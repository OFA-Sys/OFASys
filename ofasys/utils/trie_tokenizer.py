# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.


class TrieNode(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = {}
        self.is_word = False


class Trie(object):
    """
    trie树
    """

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        node = self.root
        for chars in word:  # 遍历词语中的每个字符
            child = node.data.get(chars)  # 获取该字符的子节点，
            if not child:  # 如果该字符不存在于树中
                node.data[chars] = TrieNode()  # 则创建该字符节点
            node = node.data[chars]  # 节点为当前该字符节点
        node.is_word = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for chars in word:
            node = node.data.get(chars)
            if not node:
                return False
        return node.is_word  # 判断单词是否是完整的存在在trie树中

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for chars in prefix:
            node = node.data.get(chars)
            if not node:
                return False
        return True

    def get_start(self, prefix):
        """
        Returns words started with prefix
        返回以prefix开头的所有words
        如果prefix是一个word，那么直接返回该prefix
        :param prefix:
        :return: words (list)
        """

        def get_key(pre, pre_node):
            word_list = []
            if pre_node.is_word:
                word_list.append(pre)
            for x in pre_node.data.keys():
                word_list.extend(get_key(pre + str(x), pre_node.data.get(x)))
            return word_list

        words = []
        if not self.startsWith(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for chars in prefix:
            node = node.data.get(chars)
        return get_key(prefix, node)


class TrieTokenizer(Trie):
    """
    基于字典树(Trie Tree)的中文分词算法
    """

    def __init__(self, dict_path):
        """

        :param dict_path:字典文件路径
        """
        super(TrieTokenizer, self).__init__()
        self.dict_path = dict_path
        self.create_trie_tree()
        self.punctuations = """！？｡＂＃＄％＆＇：（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."""

    def load_dict(self):
        """
        加载字典文件
        词典文件内容如下，每行是一个词：
                    AA制
                    ABC
                    ABS
                    AB制
                    AB角
        :return:
        """
        words = []
        with open(self.dict_path, mode="r", encoding="utf-8") as file:
            for line in file:
                words.append(line.strip().encode('utf-8').decode('utf-8-sig'))
        return words

    def create_trie_tree(self):
        """
        遍历词典，创建字典树
        :return:
        """
        words = self.load_dict()
        for word in words:
            self.insert(word)

    def mine_tree(self, tree, sentence, trace_index):
        """
        从句子第trace_index个字符开始遍历查找词语，返回词语占位个数
        :param tree:
        :param sentence:
        :param trace_index:
        :return:
        """
        if trace_index <= (len(sentence) - 1):
            if sentence[trace_index] in tree.data:
                trace_index = trace_index + 1
                trace_index = self.mine_tree(tree.data[sentence[trace_index - 1]], sentence, trace_index)
        return trace_index

    def tokenize(self, sentence):
        tokens = []
        sentence_len = len(sentence)
        while sentence_len != 0:
            trace_index = 0  # 从句子第一个字符开始遍历
            trace_index = self.mine_tree(self.root, sentence, trace_index)

            if trace_index == 0:  # 在字典树中没有找到以sentence[0]开头的词语
                tokens.append(sentence[0:1])  # 当前字符作为分词结果
                sentence = sentence[1 : len(sentence)]  # 重新遍历sentence
                sentence_len = len(sentence)
            else:  # 在字典树中找到了以sentence[0]开头的词语，并且trace_index为词语的结束索引
                tokens.append(sentence[0:trace_index])  # 命中词语作为分词结果
                sentence = sentence[trace_index : len(sentence)]  #
                sentence_len = len(sentence)

        return tokens


if __name__ == '__main__':
    trie_cws = TrieTokenizer('data/32w_dic.txt')

    sentence = '该方法的主要思想：词是稳定的组合，因此在上下文中，相邻的字同时出现的次数越多，就越有可能构成一个词。因此字与字相邻出现的概率或频率能较好地反映成词的可信度。'
    '可以对训练文本中相邻出现的各个字的组合的频度进行统计，计算它们之间的互现信息。互现信息体现了汉字之间结合关系的紧密程度。当紧密程 度高于某一个阈值时，'
    '便可以认为此字组可能构成了一个词。该方法又称为无字典分词。'
    tokens = trie_cws.tokenize(sentence)

    print(tokens)
