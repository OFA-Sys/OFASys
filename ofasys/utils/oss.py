# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import oss2
import requests
from tqdm import tqdm

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse


def oss_default_resource_path(local_path: str) -> str:
    path = f"oss://ofasys/{local_path}?host=oss-cn-zhangjiakou.aliyuncs.com"
    return path


def split_oss_path(url: str) -> (str, str, str):
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad oss path {}".format(url))
    bucket_name = parsed.netloc
    oss_path = parsed.path
    # Remove '/' at beginning of path.
    if oss_path.startswith("/"):
        oss_path = oss_path[1:]
    if parsed.query:
        querys = dict(k.split('=') for k in parsed.query.split('&'))
    else:
        querys = {}
    endpoint = querys.pop('host', 'oss-cn-zhangjiakou.aliyuncs.com')
    return bucket_name, oss_path, endpoint


def oss_etag(url: str) -> str:
    bucket_name, oss_path, endpoint = split_oss_path(url)
    try:
        auth = oss2.AnonymousAuth()
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        oss_object = bucket.head_object(oss_path)
        return oss_object.etag
    except oss2.exceptions.RequestError:
        return None


def oss_size(url: str) -> int:
    bucket_name, oss_path, endpoint = split_oss_path(url)
    auth = oss2.AnonymousAuth()
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    oss_object = bucket.head_object(oss_path)
    return oss_object.content_length


def oss_exists(url: str) -> bool:
    if not url.startswith('oss://'):
        return False
    bucket_name, oss_path, endpoint = split_oss_path(url)
    auth = oss2.AnonymousAuth()
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    try:
        return bucket.object_exists(oss_path)
    except:  # noqa
        return False


class TqdmUpTo(tqdm):
    def update_to(self, current, total):
        self.total = total
        self.update(current - self.n)


def oss_to_file(url: str, file_path: str, show_progress=True) -> None:
    bucket_name, oss_path, endpoint = split_oss_path(url)
    auth = oss2.AnonymousAuth()
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    url_prefix = url[: url.find('?')]
    with TqdmUpTo(
        desc=f'download {url_prefix}',
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        disable=not show_progress,
    ) as bar:
        bucket.get_object_to_file(oss_path, file_path, progress_callback=bar.update_to)
    del bucket, auth


class OssFileHandler:
    def __init__(self, bucket, oss_path, byte_range=None, retry_count=10):
        self.bucket = bucket
        self.oss_path = oss_path
        self.byte_range = byte_range
        self.retry_count = retry_count
        if byte_range is not None:
            self.index = byte_range[0]
        self.oss_connect = None

    def read(self, n=None):
        for retry_count in range(self.retry_count):
            try:
                if self.oss_connect is None:
                    self.oss_connect = self.reconnect()
                data = self.oss_connect.read(n)
            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                oss2.exceptions.RequestError,
            ) as e:
                print('oss reader ConnectionResetError {}: retry {}'.format(str(e), retry_count + 1))
                self.oss_connect = None
                continue
            if self.byte_range is not None:
                should_read = min(n, self.byte_range[1] - self.index + 1)
                if len(data) == should_read:
                    self.index += len(data)
                    return data
                print(
                    'oss reader data-length mismatch (got {} < {}): retry {}'.format(
                        len(data), should_read, retry_count + 1
                    )
                )
                self.oss_connect = None
            else:
                return data
        raise ValueError("Oss reader disconneted")

    def close(self):
        if self.oss_connect is not None and hasattr(self.oss_connect, 'close'):
            self.oss_connect.close()

    def reconnect(self):
        if self.byte_range is not None:
            return self.bucket.get_object(self.oss_path, byte_range=(self.index, self.byte_range[1]))
        else:
            return self.bucket.get_object(self.oss_path)

    def __del__(self):
        self.close()


def oss_get(url: str, byte_range=None, retry_count=10) -> oss2.models.GetObjectResult:
    """
    Fetch oss connection as file interface with retry.

    Args:
        url: oss_path in the following format: oss://bucket_name/path/xxx?host=yyy'
            host are optional
        byte_range: byte_range of file, [start, end], both inclusive
            refer to https://help.aliyun.com/document_detail/88443.html
        retry_count: the retry count of connection
    """
    bucket_name, oss_path, endpoint = split_oss_path(url)
    auth = oss2.AnonymousAuth()
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    return OssFileHandler(bucket, oss_path, byte_range, retry_count)
