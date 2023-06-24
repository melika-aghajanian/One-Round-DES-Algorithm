initial_perm = [58, 50, 42, 34, 26, 18, 10, 1,
                60, 52, 44, 36, 28, 20, 12, 4,
                62, 54, 46, 38, 30, 22, 14, 6,
                64, 56, 48, 40, 32, 24, 16, 8,
                57, 49, 41, 33, 25, 17, 9, 2,
                59, 51, 43, 35, 27, 19, 11, 3,
                61, 53, 45, 37, 29, 21, 13, 5,
                63, 55, 47, 39, 31, 23, 15, 7]

final_perm = [8, 40, 48, 16, 56, 24, 64, 32,
              39, 7, 47, 15, 55, 23, 63, 31,
              38, 6, 46, 14, 54, 22, 62, 30,
              37, 5, 45, 13, 53, 21, 61, 29,
              36, 4, 44, 12, 52, 20, 60, 28,
              35, 3, 43, 11, 51, 19, 59, 27,
              34, 2, 42, 10, 50, 18, 58, 26,
              33, 1, 41, 9, 49, 17, 57, 25]

exp_d = [32, 1, 2, 3, 4, 5, 4, 5,
         6, 7, 8, 9, 8, 9, 10, 11,
         12, 13, 12, 13, 14, 15, 16, 17,
         16, 17, 18, 19, 20, 21, 20, 21,
         22, 23, 24, 25, 24, 25, 26, 27,
         28, 29, 28, 29, 30, 31, 32, 1]

sbox = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
         [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
         [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
         [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

        [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
         [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
         [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
         [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

        [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
         [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
         [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
         [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

        [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
         [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
         [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
         [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

        [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
         [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
         [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
         [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

        [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
         [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
         [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
         [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

        [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
         [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
         [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
         [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

        [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
         [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
         [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
         [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]]

straight_p_box = [5, 25, 19, 27, 28, 11, 20, 16, 30, 14, 22, 9, 4,
                  17, 0, 15, 2, 7, 23, 13, 18, 26, 1, 8, 31, 12, 29, 3, 10, 21, 24, 6]

# --------------------------------------------------------
def hex_to_binary(hex_string):
    decimal_num = int(hex_string, 16)
    binary_str = bin(decimal_num)[2:]

    # Pad the binary string with leading zeros if necessary
    num_bits = len(hex_string) * 4
    padded_binary_str = binary_str.zfill(num_bits)

    return padded_binary_str


def apply_ip(cypher_text):
    permuted_bits = [cypher_text[i - 1] for i in initial_perm]

    return ''.join(permuted_bits)


def split(input_string):
    parts = []
    # Split the input string into 4 equal parts (each part has a length of 64) and append them to the list
    for i in range(0, len(input_string), 64):
        parts.append(input_string[i:i + 64])
    return parts


def permut(right_half):
    output_block = ""
    for i in range(len(exp_d)):
        output_block += right_half[exp_d[i] - 1]
    return output_block


def xor_binary(input1, input2):
    int_input1 = int(input1, 2)
    int_input2 = int(input2, 2)
    result = int_input1 ^ int_input2
    length = len(input1)
    binary_result = bin(result)[2:].zfill(length)

    return binary_result


def apply_s_boxes(input_bits):
    s_box_inputs = [input_bits[i:i + 6] for i in range(0, len(input_bits), 6)]
    s_box_outputs = []
    for i, s_box_input in enumerate(s_box_inputs):
        row = int(s_box_input[0] + s_box_input[5], 2)
        col = int(s_box_input[1:5], 2)
        s_box_output = sbox[i][row][col]
        s_box_outputs.append(s_box_output)
    output_bits = ''.join(format(output, '04b') for output in s_box_outputs)

    return output_bits


def straight_permutation(text):
    assert len(text) == 32, "Input must be a string of length 32"
    text2 = []
    for index in straight_p_box:
        text2.append(text[index])
    return ''.join(text2)


def final_permute(data):
    bits = [int(bit) for bit in data]
    output_bits = [bits[index - 1] for index in final_perm]
    return ''.join(str(bit) for bit in output_bits)
#------------------------------------------------------------------------
def parity_drop(key):
    # Define the positions of the bits to keep after parity drop
    keyp = [57, 49, 41, 33, 25, 17, 9,
            1, 58, 50, 42, 34, 26, 18,
            10, 2, 59, 51, 43, 35, 27,
            19, 11, 3, 60, 52, 44, 36,
            63, 55, 47, 39, 31, 23, 15,
            7, 62, 54, 46, 38, 30, 22,
            14, 6, 61, 53, 45, 37, 29,
            21, 13, 5, 28, 20, 12, 4]

    parity_dropped = ''.join([key[i - 1] for i in keyp])

    return parity_dropped


def first_round_key(key_pc1):
    shift_table = [1, 1, 2, 2,
                   2, 2, 2, 2,
                   1, 2, 2, 2,
                   2, 2, 2, 1]

    c0 = key_pc1[:28]
    d0 = key_pc1[28:]

    c_shifted = c0[shift_table[0]:] + c0[:shift_table[0]]
    d_shifted = d0[shift_table[0]:] + d0[:shift_table[0]]
    cd = c_shifted + d_shifted
    pc2_table = [14, 17, 11, 24, 1, 5,
                 3, 28, 15, 6, 21, 10,
                 23, 19, 12, 4, 26, 8,
                 16, 7, 27, 20, 13, 2,
                 41, 52, 31, 37, 47, 55,
                 30, 40, 51, 45, 33, 48,
                 44, 49, 39, 56, 34, 53,
                 46, 42, 50, 36, 29, 32]

    round_key = ''.join([cd[i - 1] for i in pc2_table])

    c0 = c_shifted
    d0 = d_shifted
    return round_key

#-----------------------------------------------------------------------
def pad_text(text):
    padding_length = 8 - (len(text) % 8)
    padding = chr(padding_length) * padding_length
    return text + padding


def remove_padding(padded_text):
    padding_length = ord(padded_text[-1])
    return padded_text[:-padding_length]

#-----------------------------------------------------------------
def text_to_bin(text):
    b = int.from_bytes(text.encode(), byteorder='big')
    binary_string = bin(b)[2:].zfill(64)
    return binary_string


def binary_to_text(binary):
    binary = binary.strip()
    if len(binary) % 8 != 0:
        raise ValueError(
            "Invalid binary string: length should be divisible by 8")
    binary_list = [binary[i:i + 8] for i in range(0, len(binary), 8)]
    text_list = [chr(int(binary_char, 2)) for binary_char in binary_list]
    text = "".join(text_list)
    return text

# -----------------------------------------------
# generating the DES key

key = '4355262724562343'

binary_key = hex_to_binary(key)

parity_dropped_key = parity_drop(binary_key)

first_round = first_round_key(parity_dropped_key)

# -----------------------------------------------
# finding the plain texts before the straight p-box as its inputs

plain_text = [
    'kootahe',
    'Zendegi',
    'Edame',
    'Dare',
    'JolotYe',
    'Daame',
    'DaemKe',
    'Mioftan',
    'Toosh',
    'HattaMo',
    'khayeSa',
    '05753jj',
    '==j95697'
]

padded_plain_text = ['', '', '', '', '', '', '', '', '', '', '', '', '']
left = ['', '', '', '', '', '', '', '', '', '', '', '', '']
right = ['', '', '', '', '', '', '', '', '', '', '', '', '']
permuted_right = ['', '', '', '', '', '', '', '', '', '', '', '', '']
xor_key = ['', '', '', '', '', '', '', '', '', '', '', '', '']
s_box_list = ['', '', '', '', '', '', '', '', '', '', '', '', '']

for i in range(len(plain_text)):
    padded_plain_text[i] = pad_text(plain_text[i])

    padded_plain_text[i] = text_to_bin(padded_plain_text[i])

    padded_plain_text[i] = apply_ip(padded_plain_text[i])

    half_index = len(padded_plain_text[i]) // 2

    left[i] = padded_plain_text[i][:half_index]

    right[i] = padded_plain_text[i][half_index:]

    permuted_right[i] = permut(right[i])

    xor_key[i] = xor_binary(permuted_right[i], first_round)

    s_box_list[i] = apply_s_boxes(xor_key[i])

    #print(s_box_list[i], '\n')

# -----------------------------------------------
# finding the cypher texts after the straight p-box as its outputs

cypher_text = ['6E2F7B25307C3144',
               'CF646E7170632D45',
               'D070257820560746',
               '5574223505051150',
               'DB2E393F61586144',
               'D175257820560746',
               'D135603D1A705746',
               'D83C6F7321752A54',
               '413A2B666D024747',
               '5974216034186B44',
               'EA29302D74463545',
               'B1203330722B7A04',
               '38693B6824232D231D1C0D0C4959590D']

c_left = ['', '', '', '', '', '', '', '', '', '', '', '', '']
c_right = ['', '', '', '', '', '', '', '', '', '', '', '', '']
xor_leftes = ['', '', '', '', '', '', '', '', '', '', '', '', '']

for i in range(len(cypher_text)):
    cypher_text[i] = hex_to_binary(cypher_text[i])

    cypher_text[i] = apply_ip(cypher_text[i])

    half_index = len(cypher_text[i]) // 2

    c_left[i] = cypher_text[i][:half_index]
    c_right[i] = cypher_text[i][half_index:]

for i in range(len(cypher_text)):
    xor_leftes[i] = xor_binary(c_left[i], left[i])

    # print(xor_leftes[i], '\n')

# -----------------------------------------------
# finding the possible positins of each element by comparing the inputs and outputs of the straight p-box

"""
res = [[]for i in range(32)]
for i in range(len(xor_leftes)):
    before = s_box_list[i]
    after = xor_leftes[i]
    for k in range(len(after)):
        ls = []
        l = 0
        while (l < len(before)):
            if (after[k] == before[l]):
                ls.append(l)
            l += 1
        res[k].append(ls)

for x in range(len(res)):
    print(res[x])
    """

# ---------------------------------------
# decrypting the given cypher text

cypher_text1 = '59346E29456A723B62354B61756D44257871650320277C741D1C0D0C4959590D'

binary_str = hex_to_binary(cypher_text1)

blocks = split(binary_str)
decoded_text = []

for block in range(len(blocks)):
    permuted_string = apply_ip(blocks[block])

    half_index = len(permuted_string) // 2

    left_half = permuted_string[:half_index]

    right_half = permuted_string[half_index:]

    permuted_right_half = permut(right_half)

    key_xor_right_half = xor_binary(permuted_right_half, first_round)

    s_box = apply_s_boxes(key_xor_right_half)

    output_str = straight_permutation(s_box)

    new_left_half = xor_binary(output_str, left_half)

    decoded_text.append(new_left_half + right_half)

    decoded_text[block] = final_permute(decoded_text[block])

    decoded_text[block] = binary_to_text(decoded_text[block])

final_text = ''

result_string = final_text.join(decoded_text)

removed_pad = remove_padding(result_string)

print(removed_pad)
