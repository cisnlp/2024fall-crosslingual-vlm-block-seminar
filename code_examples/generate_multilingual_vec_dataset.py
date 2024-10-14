from time import sleep
from typing import Union, Dict, List

from datasets import load_dataset, Dataset, DatasetDict
from googletrans import Translator as GoogleTranslator
from translate import Translator as MyMemoryTranslator
from tqdm import tqdm


def load_vec_dataset(category: str = "all"):
    data = {}
    if category not in [
        'color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness', 'all'
    ]:
        raise ValueError(
            f"Invalid category: {category}. Please choose from ['color', 'size', 'shape', 'height', "
            f"'material', 'mass', 'temperature', 'hardness', 'all']"
        )

    if category == "all":
        for task in [
            'color', 'size', 'shape', 'height', 'material', 'mass', 'temperature', 'hardness'
        ]:
            data[task] = load_dataset("tobiaslee/VEC", task)
    else:
        data[category] = load_dataset("tobiaslee/VEC", category)
    return data


def translate_func(
        text: str,
        source_language: str,
        target_language: str,
        translator=GoogleTranslator(),
        backup_translator=None,
) -> str:
    while True:
        try:
            translation = translator.translate(text,
                                               src=source_language,
                                               dest=target_language).text
            return translation
        except TypeError as te:
            print(f"Errors in translating {text}: {te}")
            print(f"Try to use the backup translator...")
            if not backup_translator:
                backup_translator = MyMemoryTranslator(to_lang=target_language)
            while True:
                try:
                    translation = backup_translator.translate(text)
                    return translation
                except Exception as e:
                    print(f"Errors in translating {text} with backup translator: {e}")
                    print(f"Retry...")
        except AttributeError as ae:
            print(f"Errors in translating {text}: {ae}")
            print(f"Retry...")


def translate_vec_dataset(
        dataset: Union[Dataset, Dict[str, Dataset], None],
        target_languages: List[str],
        concept_category: str,
        dataset_split: str = "test",
        translate_function=translate_func,
) -> Dict[str, List]:
    translated_ds = {}
    orig_ds = dataset[concept_category][dataset_split]
    for tl in target_languages:
        for i in tqdm(
                range(len(orig_ds)),
                desc=f"Translating {concept_category} to {tl}"):
            for k, v in orig_ds[i].items():
                if not isinstance(v, str) or k == "relation":
                    translated_v = v
                else:
                    translated_v = translate_function(text=v,
                                                      source_language="en",
                                                      target_language=tl)
                    sleep(0.25)
                translated_ds[k] = translated_ds.get(k, []) + [translated_v]
            translated_ds["language"] = translated_ds.get("language", []) + [tl]
    return translated_ds


def generate_multilingual_vec_dataset():
    # hf_ds = DatasetDict()
    tgt_langs = ["es", "de", "ja", "ko", "zh-CN"]
    # tgt_langs = ["es", "zh-CN"]
    vec_ds = load_vec_dataset(category="all")
    for c in vec_ds:
        if c in ["color", "size", "shape", "height", "material", "mass"]:
            continue
        translated_ds = translate_vec_dataset(
            dataset=vec_ds,
            target_languages=tgt_langs,
            concept_category=c,
        )
        hf_ds = Dataset.from_dict(translated_ds)
        hf_ds.push_to_hub("WindOcean/multilingual_vec_dataset", c, split="test")


def show_vec_dataset():
    ds = load_vec_dataset("all")
    for c in ds:
        print(ds[c]["test"][0])


if __name__ == '__main__':
    # show_vec_dataset()
    generate_multilingual_vec_dataset()
