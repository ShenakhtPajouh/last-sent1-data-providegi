import HP
import json
import numpy as np


def get_books_metadata(books_id=None):
    """

    Args:
        books_id: A list of integers, if it is not None, then it will return metadata of only these ids
    Returns:
        metadata of books. with format dictionary book_id: book_metadata. each metadata is a dictionary.

    """
    with open(HP.BOOKS_METADATA, 'r') as f:
        metadata = json.load(f)
    if books_id is not None:
        metadata = {i: metadata[i-1] for i in books_id}
    else:
        metadata = {i+1: met for i, met in enumerate(metadata)}
    return metadata


def get_books(author=None, language=None, bookshelf=None, has_text=True):
    """
    get books with specified features. authors, language and bookshelf could be str or a list of str.
    if has_text is True then return only books which have text

    Returns:
        a list of integers (book_id)

    """
    with open(HP.FEATURES_METADATA, 'r') as f:
        features_metadata = json.load(f)
    if has_text:
        books = features_metadata["has_text"]
    else:
        with open(HP.BOOKS_METADATA, 'r') as f:
            books = json.load(f)
            books = [gut["gutenberg_id"] for gut in books]
    books = set(books)

    x = author
    name_x = "authors"
    if x is not None:
        if isinstance(x, str):
            x = [x]
        x = [y for y in x if y in features_metadata[name_x]]
        bb = [features_metadata[name_x][ent] for ent in x]
        bb = sum(bb, [])
        bb = set(bb)
        books = books & bb

    x = bookshelf
    name_x = "bookshelves"
    if x is not None:
        if isinstance(x, str):
            x = [x]
        x = [y for y in x if y in features_metadata[name_x]]
        bb = [features_metadata[name_x][ent] for ent in x]
        bb = sum(bb, [])
        bb = set(bb)
        books = books & bb

    languages = features_metadata['languages']
    lans = []
    if language is not None:
        if isinstance(language, str):
            language = {language}
        else:
            language = set(language)
        for lan1 in language:
            if lan1 in languages:
                lans.append(lan1)
            for lan2 in language - {lan1}:
                l = lan1 + '/' + lan2
                if l in languages:
                    lans.append(l)
        bb = [languages[ent] for ent in lans]
        bb = sum(bb, [])
        bb = set(bb)
        books = books & bb
    return list(books)

def get_books_text(books=None):
    """
    Args:
        books: list of gutenberg book_id (int)

    Returns:
        a dictionary: {book_id: book_text}

    """
    if books is None:
        books = get_books(has_text=True)
    res = []
    for book in books:
        with open(HP.BOOKS_DIR + str(book) + ".txt", 'r') as f:
            text = f.read()
        res.append((book, text))
    return dict(res)


def get_features():
    """

    get books metadata features
    """
    with open(HP.FEATURES_METADATA, 'r') as f:
        features_metadata = json.load(f)
    return list(features_metadata.keys())


def get_keys():
    """

    get keys of paragraph metadata for each column
    a dictionary {key_name: column_number}

    """
    return dict(zip(HP.keys, range(len(HP.keys))))

def get_paragraphs_metadata(par_ids = None):
    """

    get paragraphs metadata. metadata is a numpy array of shape [paragraphs_num, paragraphs_keys]
    you can get keys of each column with get_keys()

    Args:
         par_ids: a list of paragraph global_ids. if it is not specified then it returns whole metadata

    """
    metadata = np.load(HP.PARAGRAPH_METADATA)
    if par_ids is not None:
        par_ids = np.array(par_ids) - 1
        metadata = metadata[par_ids]
    return metadata


def get_paragraphs_id(books=None, is_analysed=True, sents_num=None, words_num=None,
                      tokens_num=None, has_dialogue=None, whole_dialogue=None, output_local_id=True):
    """

    Returns all paragraphs with specified features.

    Args:
        book: a list of integers. if it is specified returned paragraphs are only in these books.
        is_analysed: if it is true then returned pre-analysed books. some paragraphs are not analysed (do not
        have sentence_num, ...) due to tokenization problem.
        sents_num: wheter an integer or a tuple of integers. if it is an integer then returns paragraphs with exact
                   specified number of sentences. if it is tuple, returns paragraphs with sentences_num in range
        words_num: like sents_num for words number.
        tokens_num: like sents_num for tokens number.
        has_dialogue: if it is True, it returns paragraphs with dialogue (`` ... '' token). if it is False returns
                      paragraphs without dialogue. if it is None, returns both of them.
        whole_dialogue: it is like has_dialogue, but checks wheter the whole paragaph is a dialogue.
        output_local_id: if it is True the output will be a dictionary of {book_id: [list of paragraphs local_id]}
                         and if it is false the output will be a list of paragraphs global id.
        Returns:
            if output_local_id is True the output will be a dictionary of {book_id: [list of paragraphs local_id]}
            and if it is false the output will be a list of paragraphs global id.


    """
    metadata = np.load(HP.PARAGRAPH_METADATA)
    keys = get_keys()

    if books is not None:
        book_id = metadata[:, keys["book_id"]]
        vec = np.zeros(shape=book_id.shape, dtype=bool)
        for bk in books:
            vec = np.logical_or(vec, book_id == bk)
        metadata = metadata[:, vec]

    vec = metadata[:, keys["is_analysed"]]
    if not is_analysed:
        vec0 = vec == 0
        not_analysed_metadata = metadata[vec0]

    vec = vec == 1
    metadata = metadata[vec]

    x = sents_num
    x_name = "sents_num"
    if x is not None:
        vec = metadata[:, keys[x_name]]
        if isinstance(x, int):
            vec = vec == x
        elif len(x) == 2:
            if not (isinstance(x[0], int) and isinstance(x[1], int)):
                raise ValueError(x_name + " must be a positive integer or tuple of two positive integers")
            vec = np.logical_and(x[0] <= vec, x[1] >= vec)
        else:
            raise ValueError(x_name + " must be a positive integer or tuple of two positive integers")
        metadata = metadata[vec]

    x = words_num
    x_name = "words_num"
    if x is not None:
        vec = metadata[:, keys[x_name]]
        if isinstance(x, int):
            vec = vec == x
        elif len(x) == 2:
            if not isinstance(x[0], int) and isinstance(x[1], int):
                raise ValueError(x_name + " must be a positive integer or tuple of two positive integers")
            vec = np.logical_and(x[0] <= vec, x[1] >= vec)
        else:
            raise ValueError(x_name + " must be a positive integer or tuple of two positive integers")
        metadata = metadata[vec]

    x = tokens_num
    x_name = "tokens_num"
    if x is not None:
        vec = metadata[:, keys[x_name]]
        if isinstance(x, int):
            vec = vec == x
        elif len(x) == 2:
            if not isinstance(x[0], int) and isinstance(x[1], int):
                raise ValueError(x_name + " must be a positive integer or tuple of two positive integers")
            vec = np.logical_and(x[0] <= vec, x[1] >= vec)
        else:
            raise ValueError(x_name + " must be a positive integer or tuple of two positive integers")
        metadata = metadata[vec]

    x = has_dialogue
    x_name = "has_dialogue"
    if x is not None:
        vec = metadata[: keys[x_name]]
        vec = vec == bool(x)
        metadata = metadata[vec]

    x = whole_dialogue
    x_name = "whole_dialogue"
    if x is not None:
        vec = metadata[: keys[x_name]]
        vec = vec == bool(x)
        metadata = metadata[vec]

    if not is_analysed:
        metadata = np.concatenate([metadata, not_analysed_metadata], 0)

    if output_local_id:
        vec = metadata[:, keys["book_id"]]
        books = list(vec)
        books = set(books)
        books = sorted(list(books))
        output = metadata[:, keys["local_id"]]
        output = {book: list(output[vec == book]) for book in books}
    else:
        output = metadata[:, keys["global_id"]]
        output = list(output)
    return output


def get_local_ids(pars):
    if isinstance(pars, dict):
        pars = sum([p for p in pars.values()], [])
    metadata = np.loadtxt(HP.PARAGRAPH_METADATA, dtype=int)
    keys = get_keys()
    pars = np.array(pars, dtype=int) - 1
    metadata = metadata[pars]
    vec = metadata[:, keys["book_id"]]
    books = list(vec)
    books = set(books)
    books = sorted(list(books))
    local_id = metadata[:, keys["local_id"]]
    local_id = {book: list(local_id[vec == book]) for book in books}
    return local_id


def get_global_ids(local_id):
    metadata = np.loadtxt(HP.PARAGRAPH_METADATA, dtype=int)
    keys = get_keys()
    vec = metadata[:, keys["book_id"]]
    global_ids = metadata[:, keys["global_id"]]
    output = dict()
    for book, pars in local_id.items():
        ids = global_ids[vec == book]
        ids = ids[np.array(pars, dtype=int) - 1]
        ids = list(ids)
        output[book] = ids
    return output


def get_paragraphs_ids_n(n, books=None, is_analysed=True, sents_num=None, words_num=None,
                         tokens_num=None, has_dialogue=None, whole_dialogue=None):
    """

    It is like get_paragraphs_ids exept that in n > 1 it returns sequential paragraphs with length n which all
    paragraphs have same features.

    Returns:
         unlike get_paragraphs_ids it only returns local_id format

    """
    assert n >= 1
    local_pars = get_paragraphs_id(books=books, is_analysed=is_analysed, sents_num=sents_num,
                                   words_num=words_num, tokens_num=tokens_num, has_dialogue=has_dialogue,
                                   whole_dialogue=whole_dialogue)
    new_local_pars = []
    for book, pars in local_pars.items():
        new_pars = []
        pars = set(pars)
        for i in pars:
            seq = tuple(i + x for x in range(n))
            if set(seq).issubset(pars):
                new_pars.append(seq)
        if len(new_pars) > 0:
            new_local_pars.append((book, new_pars))
    return dict(new_local_pars)


def get_local_global_dict(books=None):
    """

    It provides a dictionary of changing local ids to global ids for paragraphs:
        result[book_id][local_id] = global_id

    """
    if books is None:
        books = get_books()
    result = dict()
    metadata = get_paragraphs_metadata()
    keys = get_keys()
    features = np.array([keys["global_id"], keys["local_id"], keys["book_id"]])
    metadata = metadata[:, features]
    for book in books:
        met = metadata[:, np.array([1, 0])][metadata[:, 2] == book]
        met = dict(met)
        result[book] = met
    return result

def get_global_local_dict(pars=None):
    """

    Returns a dictionary for changing global ids to local ids:
        result[global_id] = (local_id, book_id)

    """
    metadata = get_paragraphs_metadata()
    keys = get_keys()
    features = np.array([keys["global_id"], keys["local_id"], keys["book_id"]])
    metadata = metadata[:, features]
    if pars is not None:
        pars = np.array(pars) - 1
        metadata = metadata[:, pars]
    return {par: (book, loc) for par, loc, book in metadata}


def get_paragraph_text(local_ids, num_sequential=1):
    """

    provide text of paragraphs. input format should be in the format of local_ids ({book_id: [list of local_ids]})
    but it can be sequential. (num_sequential > 1)

    Returns:
        1st: paragraphs text in the format of dictionary: {global_id: text}
        2nd: local_global dictionary: a dictioray which changes local_ids to global_ids:
             result[book_id][local_id] = global_id

    """
    paragraphs = list()
    local_global = get_local_global_dict(list(local_ids))
    for book, pars in local_ids.items():
        with open(HP.BOOKS_DIR + str(book) + ".txt", 'r', encoding='utf-8') as f:
            text = f.read()
        text = [p for p in text.split("\n\n") if p != ""]
        if num_sequential == 1:
            pps = set(pars)
        else:
            pps = set(sum([list(p) for p in pars], []))
        text = {p: text[p - 1] for p in pps}
        met = local_global[book]
        paragraphs = paragraphs + [(met[p], text[p]) for p in pps]
    paragraphs = dict(paragraphs)
    return paragraphs, local_global


































