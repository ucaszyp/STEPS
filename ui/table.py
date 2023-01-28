from typing import Dict
from collections import Iterable


class PyTable:
    def __init__(self, headers: list = None, title: str = 'Table'):
        """
        Initialize a table
        :param headers: header list
        """
        # init params
        self._data = {}
        self._edited = False
        self._split_lines = []
        self._line_number = 0
        self.title = title
        # other params
        self._margin = 4
        # add headers
        if headers is not None:
            for h in headers:
                self._data[h] = []

    @property
    def headers(self):
        return self._data.keys()

    def add_header(self, header: str):
        assert isinstance(header, str), 'Header must be a str.'
        assert not self._edited, 'Header can be added only before any item being inserted into table.'
        if header in self._data:
            raise ValueError('Header {} already exists.')
        self._data[header] = []

    def add_item(self, item: Dict[str, str]):
        """
        Add a line to table
        :param item: a dict contains line data
        :return:
        """
        for k in self._data:
            self._data[k].append(item[k])
        self._edited = True
        self._line_number += 1

    def add_split_line(self):
        """
        Add a split line
        :return:
        """
        self._split_lines.append(self._line_number)
        self._line_number += 1

    def column_width(self):
        result = {}
        for k, v in self._data.items():
            max_width = len(k)
            max_width = max(max_width, max([len(s) for s in v]))
            result[k] = max_width + self._margin
        return result

    def print_table(self):
        """
        Print current table
        :return:
        """
        def print_border_line():
            print('+{}+'.format('-' * (table_width - 2)))

        def print_title():
            print('|{}|'.format(self.title.center(table_width - 2)))

        def print_header():
            print('|{}|'.format('|'.join([h.center(column_width[h]) for h in self._data])))

        def print_item(num):
            print('|{}|'.format('|'.join([self._data[h][num].center(column_width[h]) for h in self._data])))

        # compute column width
        column_width = self.column_width()
        # compute table width
        table_width = sum(list(column_width.values())) + len(self._data) + 1
        # print table
        print_border_line()
        print_title()
        print_border_line()
        print_header()
        print_border_line()
        # print items
        curr_item_num = 0
        for i in range(self._line_number):
            if i in self._split_lines:
                print_border_line()
            else:
                print_item(curr_item_num)
                curr_item_num += 1
        print_border_line()
