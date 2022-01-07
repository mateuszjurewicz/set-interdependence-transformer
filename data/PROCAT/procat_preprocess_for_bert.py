"""
Small script for turning the PROCAT data files into .csv's
that can be more easily read by huggingface's datasets library.
"""


import ast
import csv
import pickle

page_break_token_in = '?PAGE_BREAK?'
page_break_token_out = 'katalog sektionspause'.upper()
fake_offer_token_in = '?NOT_REAL_OFFER?'
fake_offer_token_out = 'falsk tilbud'.upper()

# handle paths
path_catalogs = './catalog_features.csv'
path_offers = './offer_features.csv'

path_out_offer_id_to_text = './offer_id_to_text.pickle'
path_out_catalog_id_to_offer_ids = './catalog_id_to_offer_ids.pickle'
path_out_final_csv = './catalog_and_offer_texts.csv'

# get catalog id to offer text map
catalog_id_to_offer_ids = dict()

print('Loading the catalog features ...')

with open(path_catalogs, 'r', encoding='UTF-8') as catalog_csv:
    catalog_cols = ['catalog_id', 'section_ids', 'offer_ids_with_pb', 'offer_vectors_with_pb',
                    'offer_priorities_with_pb', 'num_offers', 'x', 'y']
    catalog_reader = csv.DictReader(catalog_csv, fieldnames=catalog_cols, delimiter=';')

    # skip header
    next(catalog_reader)

    for i, row_catalog in enumerate(catalog_reader):
        catalog_id = row_catalog['catalog_id']
        offer_ids = ast.literal_eval(row_catalog['offer_ids_with_pb'])
        catalog_id_to_offer_ids[catalog_id] = offer_ids

        if i % 100 == 0:
            print('{} catalog iteration ...'.format(i))

# get offer id to offer text map:
print('Loading the offer features ...')
offer_id_to_text = dict()
with open(path_offers, 'r', encoding='UTF-8') as offer_csv:
    offer_cols = ['catalog_id', 'section', 'offer_id', 'priority', 'heading', 'description', 'text', 'text_tokenized',
                  'token_length', 'offer_as_vector']
    offer_reader = csv.DictReader(offer_csv, fieldnames=offer_cols, delimiter=';')

    # skip header
    next(offer_reader)

    for j, row_offer in enumerate(offer_reader):
        offer_id = row_offer['offer_id']
        offer_text = row_offer['text']
        offer_id_to_text[offer_id] = offer_text

        if j % 1000 == 0:
            print('{} offer iteration ...'.format(j))

# persist the dictionaries
print('Saving the dictionaries...')
with open(path_out_catalog_id_to_offer_ids, 'wb') as f:
    pickle.dump(catalog_id_to_offer_ids, f)
with open(path_out_offer_id_to_text, 'wb') as f:
    pickle.dump(offer_id_to_text, f)

# reload
print('Reloading the dictionaries...')
with open(path_out_catalog_id_to_offer_ids, 'rb') as f:
    catalog_id_to_offer_ids = pickle.load(f)
with open(path_out_offer_id_to_text, 'rb') as f:
    offer_id_to_text = pickle.load(f)

# construct new csv
print('Writing to the final output csv...')
with open(path_out_final_csv, 'w', encoding='UTF-8') as f:

    # writer
    writer = csv.writer(f, delimiter=';')

    for k, (catalog_id, offer_ids) in enumerate(catalog_id_to_offer_ids.items()):

        # construct row by looking up offer text
        row = [catalog_id]
        for offer_id in offer_ids:

            # handle page breaks
            if offer_id == page_break_token_in:
                row.append(page_break_token_out)
            elif offer_id == fake_offer_token_in:
                row.append(fake_offer_token_out)
            else:
                row.append(offer_id_to_text[offer_id])

        writer.writerow(row)

        if k % 1000 == 0:
            print('{} output iteration ...'.format(k))
