import nltk
import API as api
import random as rand


def data_creator(prediction_data_set_path, multi_choice_data_set_path, paragraphs_local_id):
    prediction_data_set = open(prediction_data_set_path, "w+")
    prediction_data_set.write("book_id\t")
    prediction_data_set.write("paragraph_global_id\t")
    prediction_data_set.write("paragraph_text_without_last_sentence\t")
    prediction_data_set.write("paragraph_last_sentence\n")

    multi_choice_data_set = open(multi_choice_data_set_path, "w+")
    multi_choice_data_set.write("book_id\t")
    multi_choice_data_set.write("paragraph_global_id\t")
    multi_choice_data_set.write("paragraph_text_without_last_sentence\t")
    multi_choice_data_set.write("first_choice\t")
    multi_choice_data_set.write("second_choice\t")
    multi_choice_data_set.write("third_choice\t")
    multi_choice_data_set.write("forth_choice\t")
    multi_choice_data_set.write("index_of_correct_choice\n")

    def add_to_prediction_data_set(book_id, glob_id, temp_tokenized):
        prediction_data_set.write(str(book_id) + "\t")
        prediction_data_set.write(str(glob_id) + "\t")
        for sent in temp_tokenized[:-1]:
            prediction_data_set.write(sent.replace("\n", " ").replace("\t", " ") + " ")
        prediction_data_set.write("\t" + temp_tokenized[-1].replace("\n", " ").replace("\t", " ") + "\n")

    def add_to_multi_choice_data_set(book_id, glob_id, temp_tokenized, c, p):
        multi_choice_data_set.write(str(book_id) + "\t")
        multi_choice_data_set.write(str(glob_id) + "\t")
        for sent in temp_tokenized[:-1]:
            multi_choice_data_set.write(sent.replace("\n", " ").replace("\t", " ") + " ")

        multi_choice_data_set.write("\t")

        c.append(temp_tokenized[-1])
        c[p], c[-1] = c[-1], c[p]

        for sent in c:
            multi_choice_data_set.write(sent.replace("\n", " ").replace("\t", " ") + "\t")

        multi_choice_data_set.write(str(p) + "\n")

    def get_random(i, last_sents):
        r1 = rand.randint(0, len(last_sents) - 1)
        while r1 == i:
            r1 = rand.randint(0, len(last_sents) - 1)

        r2 = rand.randint(0, len(last_sents) - 1)
        while r2 == i | r2 == r1:
            r2 = rand.randint(0, len(last_sents) - 1)

        r3 = rand.randint(0, len(last_sents) - 1)
        while r3 == i | r3 == r1 | r3 == r2:
            r3 = rand.randint(0, len(last_sents) - 1)

        p = rand.randint(0, 3)

        return [last_sents[r1], last_sents[r2], last_sents[r3]], p

    for id, pars in paragraphs_local_id.items():
        temp_dict = {id: pars}
        text, _ = api.get_paragraph_text(temp_dict)

        last_sents = []
        tokenized_pars = []
        glob_ids = []

        for glob_id, par in text.items():
            temp_tokenized = nltk.sent_tokenize(par)
            last_sents.append(temp_tokenized[-1])
            tokenized_pars.append(temp_tokenized)
            glob_ids.append(glob_id)

        if len(glob_ids) < 4:
            continue

        for i in range(len(glob_ids)):
            add_to_prediction_data_set(id, glob_ids[i], tokenized_pars[i])

            c, p = get_random(i, last_sents)

            add_to_multi_choice_data_set(id, glob_ids[i], tokenized_pars[i], c, p)
    prediction_data_set.close()
    multi_choice_data_set.close()


if __name__ == "__main__":
    book_shelves = ["Children's Literature", "Short Stories", "Science Fiction", "Adventure", "Horror", "Fantasy",
                    "Crime Fiction"]
    book_ids = api.get_books(bookshelf=book_shelves, language="en")
    paragraphs_id = api.get_paragraphs_id(books=book_ids, sents_num=[5, 100], words_num=[30, 100000])
    prediction_data_set_path = "prediction_data_set.tsv"
    multi_choice_data_set_path = "multi_choice_data_set.tsv"
    data_creator(prediction_data_set_path, multi_choice_data_set_path, paragraphs_id)
