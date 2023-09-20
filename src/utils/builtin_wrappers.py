def contains_value(input_list, element): #nem jó
    return element in input_list

def number_of_elements_in_list(input_list):
    elemszam=len(input_list)
    return elemszam

def remove_every_element_from_list(input_list):
    input_list.clear()

def reverse_list(input_list):
    reversed_list = input_list[::-1]
    return reversed_list

def odds_from_list(input_list):
    odd_elements = [x for x in input_list if x % 2 != 0]
    return odd_elements

def number_of_odds_in_list(input_list):
    c_odd_elements = len([x for x in input_list if x % 2 != 0])
    return c_odd_elements

def contains_odd(input_list):
    for element in input_list:
        if element%2!=0:
            return True
        return False

def second_largest_in_list(input_list):
    s=sorted(set(input_list))
    return s[-2]

def sum_of_elements_in_list(input_list):
    result = sum(input_list)
    return float(result)

def cumsum_list(input_list):
    cumsum = []
    total = 0
    for element in input_list:
        total += element
        cumsum.append(total)
    return cumsum


def element_wise_sum(input_list1, input_list2):
    if len(input_list1) != len(input_list2):
        raise ValueError("A bemeneti listák hossza nem egyezik meg")

    result = [x + y for x, y in zip(input_list1, input_list2)]
    return result

def subset_of_list(input_list, start_index, end_index): #nem jó
    subset = input_list[start_index:end_index+1]
    return subset

def every_nth(input_list, step_size):
    result = input_list[::step_size]
    return result

def only_unique_in_list(input_list):
    # Az egyedi elemek számát hasonlítjuk össze a lista hosszával
    return len(set(input_list)) == len(input_list)

def keep_unique(input_list):
    unique_list = list(set(input_list))
    return unique_list


def swap(input_list, first_index, second_index):
    if first_index < 0 or first_index >= len(input_list) or second_index < 0 or second_index >= len(input_list):
        raise ValueError("Az indexek érvénytelenek")

    input_list[first_index], input_list[second_index] = input_list[second_index], input_list[first_index]
    return input_list

def remove_element_by_value(input_list, value_to_remove):
    input_list = [x for x in input_list if x != value_to_remove]
    return input_list

def remove_element_by_index(input_list, index):
    if index < 0 or index >= len(input_list):
        raise ValueError("Az index érvénytelen")

    del input_list[index]
    return input_list

def multiply_every_element(input_list, multiplier):
    result = [x * multiplier for x in input_list]
    return result

def remove_key(input_dict, key):
    if key in input_dict:
        del input_dict[key]
    return input_dict

def sort_by_key(input_dict):
    sorted_dict = {k: v for k, v in sorted(input_dict.items())}
    return sorted_dict

def sum_in_dict(input_dict):
    total = sum(input_dict.values())
    return total

def merge_two_dicts(input_dict1, input_dict2):
    merged_dict = {**input_dict1, **input_dict2}
    return merged_dict

def merge_dicts(*dicts):
    merged_dict = {}
    for d in dicts:
        merged_dict.update(d)
    return merged_dict


def sort_list_by_parity(input_list):
    even_numbers = [x for x in input_list if x % 2 == 0]
    odd_numbers = [x for x in input_list if x % 2 != 0]

    result_dict = {'even': even_numbers, 'odd': odd_numbers}
    return result_dict


def mean_by_key_value(input_dict):
    result_dict = {}

    for key, value in input_dict.items():
        if isinstance(value, list) and len(value) > 0:
            mean_value = sum(value) / len(value)
            result_dict[key] = mean_value

    return result_dict


def count_frequency(input_list):
    frequency_dict = {}

    for item in input_list:
        if item in frequency_dict:
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1

    return frequency_dict


