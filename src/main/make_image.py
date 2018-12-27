import cv2
import numpy as np


def letter_only(aa):
    return int((ord(aa) - 65) / 26 * 255)


def dash_contained(aa):
    if aa == '-':
        return 0
    else:
        return 55 + int((ord(aa) - 65) / 26 * 200)


def monochrome(source_seq, size, output_file, content_type='letter_only'):
    empty = np.zeros((size, size, 3), dtype=np.int16)

    for index, aa in enumerate(source_seq):
        row = index // size
        column = index % size
        if content_type == 'letter_only':
            num = letter_only(aa)
        else:
            num = dash_contained(aa)
        empty[row][column] = (num, num, num)

    cv2.imwrite(output_file, empty)


def fixed_size(source_seq, size, output_file, content_type='letter_only'):
    size = 28
    empty = np.zeros((size, size, 3), dtype=np.int16)

    seq_len = len(source_seq)

    if seq_len > 28 * 28:
        redun_half = (seq_len - 28 * 28) // 2
        cut_start = redun_half
        cut_end = cut_start + 784
        source_seq = source_seq[cut_start:cut_end]
    elif seq_len < 28 * 28:
        orig_seq = source_seq
        index = 0
        while len(source_seq) < 28 * 28:
            source_seq += orig_seq[index % len(orig_seq)]
            index += 1

    for index, aa in enumerate(source_seq):
        row = index // size
        column = index % size
        if content_type == 'letter_only':
            num = letter_only(aa)
        else:
            num = dash_contained(aa)
        empty[row][column] = (num, num, num)

    cv2.imwrite(output_file, empty)


def resized_mono(source_seq, size, output_file, content_type='letter_only'):
    empty = np.zeros((size, size, 3), dtype=np.int16)

    for index, aa in enumerate(source_seq):
        row = index // size
        column = index % size
        if content_type == 'letter_only':
            num = letter_only(aa)
        else:
            num = dash_contained(aa)
        empty[row][column] = (num, num, num)

    resized = cv2.resize(empty, (128, 128))
    # resized = empty
    cv2.imwrite(output_file, resized)


if __name__ == '__main__':
    source_seq = 'MNESKPGDSQNLACVFCRKHDDCPNKYGEKKTKEKWNLTVHYYCLLMSSGIWQRGKEEEGVYGFLIEDIRKEVNRASKLKCCVCKKNGASIGCVAPRCKRSYH' \
                 'FPCGLQRECIFQFTGNFASFCWDHRPVQIITSNNYRESLPCTICLEFIEPIPSYNILRSPCCKNAWFHRDCLQVQAINAGVFFFRCTICNNSDIFQKEMLRMG' \
                 'IHIPEKDASWELEENAYQELLQHYERCDVRRCRCKEGRDYNAPDSKWEIKRCQCCGSSGTHLACSSLRSWEQNWECLECRGIIYNSGEFQKAKKHVLPNSNNV' \
                 'GITDCLLEESSPKLPRQSPGSQSKDLLRQGSKFRRNVSTLLIELGFQIKKKTKRLYINKANIWNSALDAFRNRNFNPSYAIEVAYVIENDNFGSEHPGSKQEF' \
                 'LSLLMQHLENSSLFEGSLSKNLSLNSQALKENLYYEAGKMLAISLVHGGPSPGFFSKTLFNCLVYGPENTQPILDDVSDFDVAQIIIRINTATTVADLKSIIN' \
                 'ECYNYLELIGCLRLITTLSDKYMLVKDILGYHVIQRVHTPFESFKQGLKTLGVLEKIQAYPEAFCSILCHKPESLSAKILSELFTVHTLPDVKALGFWNSYLQ' \
                 'AVEDGKSTTTMEDILIFATGCSSIPPAGFKPTPSIECLHVDFPVGNKCNNCLAIPITNTYKEFQENMDFTIRNTLRLEKEESSHYIGH'

    fixed_size(source_seq, 28, './data/fasta/helix/empty.png')
