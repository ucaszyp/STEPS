import os


# comment char
_COMMENT_CHAR = '#'


def read_list_from_file(fn: str, items_per_line=None, separator=None, comment_char=_COMMENT_CHAR):
    """
    Read a list from given file
    Note: list file is organized by lines, comments are started with _COMMENT_CHAR
    :param fn: file name
    :param items_per_line: number of items per line
    :param separator: separator. Using space char when None
    :param comment_char: char indicating comment
    :return:
    """
    def get_items(src_line: str):
        if items_per_line is None:
            items = src_line.split(separator)
        else:
            items = src_line.split(separator, items_per_line)[: items_per_line]
        return items[0] if len(items) == 1 else items

    # check parameters
    assert os.path.exists(fn)
    assert (items_per_line is None) or (items_per_line > 0)
    # store result
    result = []
    # read file
    with open(fn, 'r') as f:
        # handle lines
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith(comment_char):
                continue
            result.append(get_items(line))
    # return result
    return result
