#X8Z8HR


#1
def evens_from_list(input_list):
    even_elements = [x for x in input_list if x % 2 == 0]
    return even_elements
#2
def every_element_is_odd(input_list):
    for element in input_list:
        if element % 2 == 0:
            return False
    return True

#3
def kth_largest_in_list(input_list,kth_largest):
    s=sorted(set(input_list))
    return s[-kth_largest]

#4
def cumavg_list(input_list):
    cumulative_sum = 0
    cumavg_values = []

    for i, value in enumerate(input_list, start=1):
        cumulative_sum += value
        cumavg = cumulative_sum / i
        cumavg_values.append(cumavg)

    return cumavg_values

#5
def element_wise_multiplication(input_list1, input_list2):
    if len(input_list1) != len(input_list2):
        raise ValueError("A bemeneti listák hossza nem egyezik meg")

    result = [x * y for x, y in zip(input_list1, input_list2)]
    return result

#6
def merge_lists(*lists):
    merged_list = []

    for lst in lists:
        merged_list.extend(lst)

    return merged_list

#7
def squared_odds(input_list):
    result = [x ** 2 for x in input_list if x % 2 != 0]

    return result

#8
def reverse_sort_by_key(input_dict):
    sorted_dict = dict(sorted(input_dict.items(), key=lambda item: item[0], reverse=True))
    return sorted_dict

#9
def sort_list_by_divisibility(input_list):
    result_dict = {
        'by_two': [],
        'by_five': [],
        'by_two_and_five': [],
        'by_none': []
    }

    for num in input_list:
        if num % 2 == 0 and num % 5 == 0:
            result_dict['by_two_and_five'].append(num)
        elif num % 2 == 0:
            result_dict['by_two'].append(num)
        elif num % 5 == 0:
            result_dict['by_five'].append(num)
        else:
            result_dict['by_none'].append(num)

    return result_dict