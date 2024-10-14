from datasets import load_dataset


def load_multilingual_vec_dataset(category: str = "all"):
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
            data[task] = load_dataset("WindOcean/multilingual_vec_dataset", task)
    else:
        data[category] = load_dataset("WindOcean/multilingual_vec_dataset", category)
    return data


def show_multilingual_vec_dataset():
    dataset = load_multilingual_vec_dataset("all")
    print(dataset['material']['test'][0])
    # {'obj': 'silla', 'positive': 'madera', 'negative': 'jade', 'relation': 'material', 'language': 'es'}


def eval_models_on_multilingual_vec():
    # An example, please implement all the functions by yourself
    dataset = load_multilingual_vec_dataset("all")
    for category in dataset:
        dataset_size = len(dataset[category]['test'])
        for i in range(dataset_size):
            prompt = get_prompt_for_vec(category=category,
                                        language=dataset[category]['test'][i]['language'])
            answer = run_model(model_name, prompt, call_method="vllm")
            evaluation_results = evaluate(answer, ground_truth)


if __name__ == '__main__':
    show_multilingual_vec_dataset()
