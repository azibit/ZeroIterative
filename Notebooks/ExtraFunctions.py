import os
import random
from pathlib import Path
import math, base64
from SingleClassPrediction import SingleClassImageTask
from MultiClassPrediction import MultiClassImageTask
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from ImageAugmenter import ImageAugmenter

# Iterate over the number of experiments
def log_experiment_results(dataset_classes_of_interest, exp_results, experiment, model_name, model_type, number_of_classes_to_predict, image_paths, classes_of_interest_list, voting_result_list=None):
    image_paths_classes = [dataset_classes_of_interest[image_path.split("\\")[-2]] for image_path in image_paths]

    print(f"Predicted Classes of Interest: {classes_of_interest_list}")
    print(f"Correct Classes: {image_paths_classes}")
    curr_pred_acc = compute_top_n_accuracy(image_paths_classes, classes_of_interest_list)
    print(f"Accuracy: {curr_pred_acc}")

    voting_result_dict = {}
    voting_preds = None
    if voting_result_list:
        for item in voting_result_list:
            voting_result_dict[item.image_path] = item.classes

        voting_preds = [voting_result_dict[image_path] for image_path in image_paths]

    # Append the results
    exp_results.append({
                "experiment": experiment,
                "number of classes": number_of_classes_to_predict,
                "model_name": model_name,
                "model_type": model_type,
                "accuracy": curr_pred_acc,
                "image_paths": image_paths,
                "LLM Predicted Classes": classes_of_interest_list,
                "Correct Classes": image_paths_classes,
                "Voting Result": voting_preds
            })

def create_new_multitask_image_tasks(number_of_classes_to_predict, result_list):
    new_multitask_image_tasks = []
    for item in result_list:
        new_task = MultiClassImageTask(item.get_image_path(), item.get_classes(), number_of_classes_to_predict, item.get_encoded_image(), item.get_features())
        new_task.set_image_textual_description(item.get_image_textual_description())
        new_multitask_image_tasks.append(new_task)
    return new_multitask_image_tasks

def transform_result_list_to_single_image_tasks(result_list):
    _class_dictionary = {}
    _feature_dictionary = {}
    _image_textual_description_dictionary = {}

    for item in result_list:
        if item.image_path not in _class_dictionary:
            _class_dictionary[item.image_path] = item.classes.copy()
            _feature_dictionary[item.image_path] = item.features.copy()
            _image_textual_description_dictionary[item.image_path] = item.image_textual_description
        else:
            _class_dictionary[item.image_path].extend(item.classes)
            _feature_dictionary[item.image_path].extend(item.features)
            # _image_textual_description_dictionary[item.image_path].extend(item.image_textual_description)

    # Create a new list of SingleClassImageTasks
    new_single_class_image_tasks = []
    new_single_class_most_predicted_results = []
    for key, value in _class_dictionary.items():
        new_single_class_image_tasks.append(SingleClassImageTask(key, list(set(value)), _feature_dictionary[key], image_textual_description=_image_textual_description_dictionary[key]))
        new_single_class_most_predicted_results.append(SingleClassImageTask(key, [item[0] for item in Counter(value).most_common()],  _feature_dictionary[key]))

    return new_single_class_image_tasks, new_single_class_most_predicted_results

def get_random_image_variations(augmenter, num_samples=1):
    """
    Creates a set of image variations and returns random samples.
    
    Args:
        augmenter: Image augmenter object
        num_samples (int): Number of random variations to return (default: 1)
        
    Returns:
        list: List of randomly selected image variations
    """
    variations = [
        augmenter.rotate(90).flip_horizontal().adjust_brightness(1.2).encode(),  # Variation 1
        augmenter.rotate(180).flip_horizontal().adjust_brightness(0.8).encode(), # Variation 2
        augmenter.rotate(270).flip_horizontal().adjust_brightness(1.1).encode(), # Variation 3
        augmenter.rotate(45).flip_horizontal().adjust_brightness(0.9).encode()   # Variation 4
    ]
    
    # Ensure we don't try to sample more variations than available
    num_samples = min(num_samples, len(variations))
    
    # Return random sample(s)
    return random.sample(variations, num_samples)

# Get the image paths for each image path
def create_multiclass_augmented_image_tasks(image_paths, classes_of_interest_list, number_of_classes_to_predict, total_number_of_augmentations = 1, 
                                            consistency_sample = 2, dictionary_of_interest = None, not_of_interest = False):
    
    multiclass_image_tasks = []
    total_number_of_augmentations = total_number_of_augmentations - 1

    for image_path, classes_of_interest_per_img in zip(image_paths, classes_of_interest_list):
        if not_of_interest:
            classes_of_interest_per_img_pre = [x for x in dictionary_of_interest[image_path] if x not in classes_of_interest_per_img]
            classes_of_interest_per_img.extend(classes_of_interest_per_img_pre[0:consistency_sample])
        
        random.shuffle(classes_of_interest_per_img)

        augmenter = ImageAugmenter(image_path=image_path)
        task = MultiClassImageTask(image_path, classes_of_interest_per_img, number_of_classes_to_predict, augmenter.encode())
        multiclass_image_tasks.append(task)


        if total_number_of_augmentations == 0:
            continue
        image_variations = get_random_image_variations(augmenter, num_samples=total_number_of_augmentations)
        for encoded_result in image_variations:
            classes_of_interest_per_img = random.sample(classes_of_interest_per_img, len(classes_of_interest_per_img))
            task_variation = MultiClassImageTask(image_path, classes_of_interest_per_img, number_of_classes_to_predict, encoded_result)
            multiclass_image_tasks.append(task_variation)

    random.shuffle(multiclass_image_tasks)
    return multiclass_image_tasks

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def compute_top_n_accuracy(image_class_name_list, classes_of_interest_list):
   """
   Compute accuracy where:
   image_class_name_list: Always a single list ['class1', 'class2']
   classes_of_interest_list: Can be either ['class1', 'class2'] or [['class1', 'class2'], ['class3', 'class4']]
   
   Returns:
   str: Format "X out of Y"
   """
   if len(image_class_name_list) != len(classes_of_interest_list):
       raise ValueError("Number of images and class lists must match")
   
   if not image_class_name_list:
       return "0 out of 0"
   
   correct_predictions = 0
   
   # Check if classes_of_interest is a list of lists
   if isinstance(classes_of_interest_list[0], list):
       for true_class, predicted_classes in zip(image_class_name_list, classes_of_interest_list):
           if true_class in predicted_classes:
               correct_predictions += 1
   else:
       # Single list case - match by position
       for true_class, predicted_class in zip(image_class_name_list, classes_of_interest_list):
           if true_class == predicted_class:
               correct_predictions += 1
   
   return f"{correct_predictions} out of {len(image_class_name_list)}"

# Get the image paths for each image path
def create_singleclass_image_tasks(image_paths, classes_of_interest_list):
    singleclass_image_tasks = []
    for image_path, classes_of_interest_per_img in zip(image_paths, classes_of_interest_list):
        image_task = SingleClassImageTask(image_path, classes_of_interest_per_img)
        singleclass_image_tasks.append(image_task)

    return singleclass_image_tasks

# Get the image paths for each image path
def create_multiclass_image_tasks(image_paths, classes_of_interest_list, number_of_classes_to_predict, consistency_sample = 2, dictionary_of_interest = None, not_of_interest = False):
    multiclass_image_tasks = []
    for image_path, classes_of_interest_per_img in zip(image_paths, classes_of_interest_list):
        random.shuffle(classes_of_interest_per_img)
        if not_of_interest:
            classes_of_interest_per_img_pre = [x for x in dictionary_of_interest[image_path] if x not in classes_of_interest_per_img]
            # classes_of_interest_per_img_pre.extend(classes_of_interest_per_img[0:consistency_sample])
            # random.shuffle(classes_of_interest_per_img_pre)
            # classes_of_interest_per_img = classes_of_interest_per_img_pre
            classes_of_interest_per_img.extend(classes_of_interest_per_img_pre[0:consistency_sample])
        image_task = MultiClassImageTask(image_path, classes_of_interest_per_img, number_of_classes_to_predict)
        multiclass_image_tasks.append(image_task)

    return multiclass_image_tasks

def extract_image_classes_from_multiclass_predictions_with_augmentation_and_features(rs, number_of_classes = None):
    result_list = []
    image_paths = []
    classes_list = []
    features_list = []

    right_number_of_classes_for_next_step = 0

    image_path_dict = {}
    image_path_features_dict = {}
    for image in rs['images']:
        image_path = image['image_path']
        predicted_classes = image['predicted_classes']

        right_number_of_classes_for_next_step = len(predicted_classes)

        if image_path not in image_path_dict:
            image_path_dict[image_path] = []
        # else:
        class_list = []
        feature_list = []
        for predicted_class in predicted_classes:
            class_list.append(predicted_class['class'])
            feature_list.extend(predicted_class['key_features'])

        image_path_dict[image_path].extend(class_list)
        image_path_features_dict[image_path] = feature_list

    for image_path, class_list in image_path_dict.items():
        class_list = list(set(class_list))
        # class_list = Counter(class_list) 
        # class_list = [x[0] for x in class_list.most_common(right_number_of_classes_for_next_step)]
        result_list.append(
            SingleClassImageTask(image_path=image_path, classes=class_list) 
            if not number_of_classes 
            else MultiClassImageTask(image_path=image_path, classes=class_list, num_predictions=number_of_classes)
        )
        image_paths.append(image_path)
        classes_list.append(class_list)
        
    return result_list, image_paths, classes_list, image_path_features_dict

def extract_image_classes_from_multiclass_predictions_with_augmentation(rs, number_of_classes = None):
    result_list = []
    image_paths = []
    classes_list = []

    right_number_of_classes_for_next_step = 0

    image_path_dict = {}
    for image in rs['images']:
        image_path = image['image_path']
        predicted_classes = image['predicted_classes']

        right_number_of_classes_for_next_step = len(predicted_classes)

        if image_path not in image_path_dict:
            image_path_dict[image_path] = []
        # else:
        class_list = []
        for predicted_class in predicted_classes:
            class_list.append(predicted_class['class'])
        image_path_dict[image_path].extend(class_list)

    for image_path, class_list in image_path_dict.items():
        class_list = list(set(class_list))
        # class_list = Counter(class_list) 
        # class_list = [x[0] for x in class_list.most_common(right_number_of_classes_for_next_step)]
        result_list.append(
            SingleClassImageTask(image_path=image_path, classes=class_list) 
            if not number_of_classes 
            else MultiClassImageTask(image_path=image_path, classes=class_list, num_predictions=number_of_classes)
        )
        image_paths.append(image_path)
        classes_list.append(class_list)
        
    return result_list, image_paths, classes_list

def extract_image_classes_from_multiclass_predictions(rs, number_of_classes = None):
    result_list = []
    image_paths = []
    classes_list = []
    for image in rs['images']:
        image_path = image['image_path']
        predicted_classes = image['predicted_classes']
        encoded_image = image['encoded_image'] if 'encoded_image' in image else None
        description = image['image_textual_description'] if 'image_textual_description' in image else None

        class_list = []
        features_list = []
        for predicted_class in predicted_classes:
            class_list.append(predicted_class['class'])
            features_list.extend(predicted_class['key_features'])

        # Shuffle the classes
        class_list = random.sample(class_list, len(class_list))
        features_list = list(set(random.sample(features_list, len(features_list))))

        result_list.append(
            SingleClassImageTask(image_path=image_path, classes=class_list, features=features_list, image_textual_description=description) 
            if not number_of_classes 
            else MultiClassImageTask(image_path=image_path, classes=class_list, num_predictions=number_of_classes, encoded_image=encoded_image, features=features_list, image_textual_description=description)
        )

        image_paths.append(image_path)
        classes_list.append(class_list)
        
    return result_list, image_paths, classes_list

def extract_single_classes(f_res, dataset_classes_of_interest):
    image_paths = []
    correct_classes = []
    pred_classes = []
    for final_img_pred in f_res['images']:
        img_path = final_img_pred['image_path']
        image_paths.append(img_path)
        correct_classes.append(dataset_classes_of_interest[img_path.split('\\')[-2]])
        pred_classes.append(final_img_pred['predicted_classes'][0]['class'])

    return correct_classes, pred_classes, image_paths

def count_matches(list1, list2):
    return str(sum(1 for a, b in zip(list1, list2) if a == b))  + " out of " + str(len(list2))  

def get_balanced_random_images(
    root_dir: str, 
    n: int, 
    seed = 42
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Selects exactly n random images from subdirectories, distributing them as evenly as possible.
    The selection is repeatable when using the same seed value.
    
    Args:
        root_dir (str): Path to root directory containing image folders
        n (int): Exact total number of images to select across all folders
        seed (int | None): Random seed for reproducibility. If None, selection will be random.
        
    Returns:
        Tuple[Dict[str, List[str]], List[str]]: 
            - Dictionary mapping folder names to lists of selected image paths
            - Flat list of all selected image paths
    
    Raises:
        ValueError: If n is larger than available images or folders are empty
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Get all subdirectories
    subdirs = sorted([d for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))])
    
    if not subdirs:
        raise ValueError("No subdirectories found in root directory")
    
    # Calculate base number of images per folder and remainder
    base_images_per_folder = n // len(subdirs)
    remainder = n % len(subdirs)
    
    if base_images_per_folder < 0:
        raise ValueError(f"Cannot select {n} total images from {len(subdirs)} folders")
    
    result = {}
    result_list = []
    total_selected = 0
    
    # Get all image files from each subfolder
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    
    # First, collect all valid images from each directory
    folder_images = {}
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        image_files = sorted([
            f for f in os.listdir(subdir_path)
            if os.path.isfile(os.path.join(subdir_path, f)) and
            Path(f).suffix.lower() in valid_extensions
        ])
        folder_images[subdir] = image_files
    
    # Distribute images ensuring we get exactly n images
    for i, subdir in enumerate(subdirs):
        subdir_path = os.path.join(root_dir, subdir)
        
        # Calculate how many images to take from this folder
        images_needed = base_images_per_folder
        if i < remainder:  # Distribute remainder one extra per folder until used up
            images_needed += 1
            
        if len(folder_images[subdir]) < images_needed:
            raise ValueError(
                f"Folder {subdir} has fewer than {images_needed} images "
                f"(has {len(folder_images[subdir])})"
            )
        
        # Randomly select the required number of images
        selected_images = random.sample(folder_images[subdir], images_needed)
        total_selected += len(selected_images)
        
        # Store full paths to selected images
        result[subdir] = [os.path.join(subdir_path, img) for img in selected_images]
        result_list.extend(result[subdir])
    
    assert total_selected == n, f"Internal error: Selected {total_selected} images instead of {n}"
    
    # Reset random seed to avoid affecting other code
    if seed is not None:
        random.seed()
        
    return result, result_list