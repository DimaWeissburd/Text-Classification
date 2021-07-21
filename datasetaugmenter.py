import nlpaug.augmenter.word as naw
from transformers import MarianMTModel, MarianTokenizer
from utility import clean_string, remove_duplicate_lines

def download(model_name):
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model

def translate(texts, model, tokenizer, language):
    formatter_fn = lambda txt: f'{txt}' if language == 'en' else f'>>{language}<< {txt}'
    original_texts = [formatter_fn(txt) for txt in texts]
    tokens = tokenizer.prepare_seq2seq_batch(original_texts, return_tensors='pt')
    translated = model.generate(**tokens)
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts

def back_translate(texts, language_src, language_dst, language_group):
    tmp_lang_tokenizer, tmp_lang_model = download('Helsinki-NLP/opus-mt-en-' + language_group)
    src_lang_tokenizer, src_lang_model = download('Helsinki-NLP/opus-mt-' + language_group + '-en')
    translated = translate(texts, tmp_lang_model, tmp_lang_tokenizer, language_dst)
    back_translated = translate(translated, src_lang_model, src_lang_tokenizer, language_src)
    return back_translated

def read_dataset(path):
    f = open(path, 'r', encoding='utf-8')
    sentences = []
    labels = []
    for line in f:
        content = line.replace('\n', '').split('\t')
        sentences.append(content[0])
        labels.append(content[1])
    f.close()
    return sentences, labels

def write_new_dataset(path, originaltexts, augtexts, labels):
    f = open(path, 'w', encoding='utf-8')
    for i, line in enumerate(originaltexts):
        f.write(originaltexts[i] + '\t' + labels[i] + '\n')
        if originaltexts[i] != augtexts[i]:
            f.write(augtexts[i] + '\t' + labels[i] + '\n')
    f.close()

def augment(input_path, output_path, technique, target_language='fr', language_group='ROMANCE'):
    originaltexts, labels = read_dataset(input_path)
    if technique == 'back-translation':
        augtexts = back_translate(originaltexts, 'en', target_language, language_group)
    if technique == 'synonyms':
        aug = naw.SynonymAug()
        augtexts = []
        for sentence in originaltexts:
            augtexts.append(aug.augment(sentence))
    write_new_dataset(output_path, originaltexts, augtexts, labels)

def main():
    augment('Data/Train.txt', 'Data/AugmentedTrain.txt', 'synonyms')
    #augment('Data/Train.txt', 'Data/AugmentedTrain.txt', 'back-translation', 'fr', 'ROMANCE')
    #augment('Data/Train.txt', 'Data/AugmentedTrain.txt', 'back-translation', 'yue', 'zh')
    #remove_duplicate_lines('Data/Train.txt', 'Data/TrainWithoutDuplicates.txt')

if __name__ == '__main__':
    main() 