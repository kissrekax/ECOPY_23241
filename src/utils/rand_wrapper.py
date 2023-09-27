import random

def random_from_list(input_list):
    if not input_list:
        raise ValueError("Az input_list üres, nincs elem kiválasztva.")

    random_element = random.choice(input_list)
    return random_element

def random_sublist_from_list(input_list, number_of_elements):
    if not input_list:
        raise ValueError("Az input_list üres, nincs elem kiválasztva.")

    if number_of_elements <= 0:
        raise ValueError("A number_of_elements értéke nem pozitív.")

    if number_of_elements > len(input_list):
        raise ValueError("A number_of_elements nagyobb, mint az input_list hossza.")

    random_sublist = random.sample(input_list, number_of_elements)
    return random_sublist

def random_from_string(input_string):
    if not input_string:
        raise ValueError("Az input_string üres, nincs elem kiválasztva.")

    random_char = random.choice(input_string)
    return random_char

def hundred_small_random():
    random_list = [random.randint(0, 1) for _ in range(100)]
    return random_list

def hundred_large_random():
    random_list = [random.randint(10, 1000) for _ in range(100)]
    return random_list


def five_random_number_div_three(): #csak így áll mindig 5 elemből a lista
    random_list = []
    while len(random_list) < 5:
        number = random.randint(9, 1000)
        if number % 3 == 0:
            random_list.append(number)
    return random_list

def random_reorder(input_list):
    if not input_list:
        return input_list  # Ha az input_list üres, nem kell összekeverni.

    shuffled_list = input_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list

def uniform_one_to_five():
    return random.uniform(1, 5)
