"""
Utilities for working with the local dataset cache.
This file is adapted from `AllenNLP <https://github.com/allenai/allennlp>`_.
and `huggingface <https://github.com/huggingface>`_.
"""

# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import fnmatch
import json
import logging
import os
import shutil
import tarfile
import tempfile
from functools import partial, wraps
from hashlib import sha256
from io import open
from pathlib import Path

import torch
from datasets.utils.filelock import FileLock

from .oss import oss_etag, oss_to_file

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse


OFA_CACHE_HOME = os.path.expanduser(os.getenv("OFA_CACHE_HOME", "~/.cache/ofa"))

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def load_archive_file(archive_file):
    # redirect to the cache, if necessary
    try:
        resolved_archive_file = cached_path(archive_file, cache_dir=None)
    except EnvironmentError:
        logger.info(
            "Archive name '{}' was not found in archive name list. "
            "We assumed '{}' was a path or URL but couldn't find any file "
            "associated to this path or URL.".format(
                archive_file,
                archive_file,
            )
        )
        return None

    if resolved_archive_file == archive_file:
        logger.info("loading archive file {}".format(archive_file))
    else:
        logger.info("loading archive file {} from cache at {}".format(archive_file, resolved_archive_file))

    # Extract archive to temp dir and replace .tar.bz2 if necessary
    tempdir = None
    if not os.path.isdir(resolved_archive_file):
        tempdir = tempfile.mkdtemp()
        logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
        ext = os.path.splitext(archive_file)[1][1:]
        with tarfile.open(resolved_archive_file, "r:" + ext) as archive:
            top_dir = os.path.commonprefix(archive.getnames())
            archive.extractall(tempdir)
        os.remove(resolved_archive_file)
        shutil.move(os.path.join(tempdir, top_dir), resolved_archive_file)
        shutil.rmtree(tempdir)

    return resolved_archive_file


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the URL's, delimited
    by a period.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = OFA_CACHE_HOME
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def cached_path_from_pm(url_or_filename):
    """
    Tries to cache the specified URL using PathManager class.
    Returns the cached path if success otherwise failure.
    """
    try:
        from ofasys.utils.file_io import PathManager

        local_path = PathManager.get_local_path(url_or_filename)
        return local_path
    except Exception:
        return None


def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = OFA_CACHE_HOME
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https", "s3", "oss"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        cached_path = cached_path_from_pm(url_or_filename)
        if cached_path:
            return cached_path
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        from botocore.exceptions import ClientError

        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url):
    """Check ETag on S3 object."""
    import boto3

    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file):
    """Pull a file directly from S3."""
    import boto3

    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def request_wrap_timeout(func, url):
    import requests

    for attempt, timeout in enumerate([10, 20, 40, 60, 60]):
        try:
            return func(timeout=timeout)
        except requests.exceptions.Timeout as e:
            logger.warning(
                "Request for %s timed-out (attempt %d). Retrying with %d secs",
                url,
                attempt,
                timeout,
                exc_info=e,
            )
            continue
    raise RuntimeError(f"Unable to fetch file {url}")


def http_get(url, temp_file):
    import requests
    from tqdm import tqdm

    req = request_wrap_timeout(partial(requests.get, url, stream=True), url)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def local_file_lock(file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    def lock_wrap(func):  # noqa
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            with FileLock(file_name) as lock:
                ret = func(*args, **kwargs)
            return ret
        return wrapped_function

    return lock_wrap


@local_file_lock(os.path.join(OFA_CACHE_HOME, 'cache.lock'))
def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = OFA_CACHE_HOME
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    elif url.startswith("oss://"):
        etag = oss_etag(url)
    else:
        try:
            import requests

            response = request_wrap_timeout(partial(requests.head, url, allow_redirects=True), url)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get("ETag")
        except RuntimeError:
            etag = None

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # If we don't have a connection (etag is None) and can't identify the file
    # try to get the last downloaded one
    if not os.path.exists(cache_path) and etag is None:
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + ".*")
        matching_files = list(filter(lambda s: not s.endswith(".json"), matching_files))
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile(dir=OFA_CACHE_HOME) as temp_file:
            # logger.info(
            #     "%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("oss://"):
                oss_to_file(url, temp_file.name)
            elif url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            # logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            # logger.info("creating metadata file for %s", cache_path)
            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w") as meta_file:
                output_string = json.dumps(meta)
                meta_file.write(output_string)

            # logger.info("removing temp file %s", temp_file.name)

    return cache_path


def read_set_from_file(filename):
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, "r", encoding="utf-8") as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path, dot=True, lower=True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


@local_file_lock(os.path.join(OFA_CACHE_HOME, 'cache.lock'))
def download_and_unzip(database_path, relative_path=""):
    zip_filename = database_path.split("/")[-1]
    localpath = os.path.join(OFA_CACHE_HOME, relative_path, zip_filename)
    os.makedirs(os.path.join(OFA_CACHE_HOME, relative_path), exist_ok=True)
    if not os.path.exists(localpath):
        oss_to_file(database_path, localpath)
    if not os.path.exists(os.path.join(OFA_CACHE_HOME, relative_path, zip_filename.split(".")[0])):
        os.system("unzip -o {} -d {}".format(localpath, os.path.join(OFA_CACHE_HOME, relative_path)))
    return os.path.join(OFA_CACHE_HOME, relative_path)
