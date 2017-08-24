#!/usr/bin/env python

# -*- coding: utf-8 -*-

import argparse
import sys

import chainer
import chainer.functions as F

import numpy as np

import net
from shogi import Shogi


def main():
    parser = argparse.ArgumentParser(description='CNN Shogi USI Interface:')
    parser.add_argument('--npz', '-n', default='result/snapshot',
                        help='Snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    xp = chainer.cuda.cupy if args.gpu >= 0 else np

    while True:
        try:
            if sys.version_info[0] == 2:
                command = raw_input()
            else:
                command = input()
        except EOFError:
            continue
        command = command.split()
        if command[0] == 'usi':
            print('id name CNNShogi')
            print('id author not522 modified by air-h-128k-il')
            print('usiok')
        elif command[0] == 'isready':
            model = net.Model()
            chainer.serializers.load_npz(args.npz, model)
            print('readyok')
        elif command[0] == 'position':
            shogi = Shogi()
            for move in command[1:]:
                if move == 'startpos' or move == 'moves':
                    continue
                shogi.move(move)
        elif command[0] == 'go':
            res = model(np.asarray([shogi.get_channels()]))
            res = F.argmax(res)
            res = res.data
            channel = res / 81
            to_y = res % 81 / 9
            to_x = res % 9
            if shogi.turn == 1:
                shogi.board = xp.flipud(shogi.board)
                shogi.board = xp.fliplr(shogi.board)
            if channel < 20:
                dy = [0, 0, 1, -1, 1, -1, 1, -1, 2, 2]#移動方向らしい
                dx = [1, -1, 0, 0, 1, -1, -1, 1, 1, -1]#移動方向らしい
                try:
                    for i in range(1, 9):
                        from_y = (to_y).astype(int) + i * dy[(channel % 10).astype(int)]
                        from_x = (to_x).astype(int) + i * dx[(channel % 10).astype(int)]
                        if shogi.board[from_y][from_x] != 0:
                            break
                except IndexError:
                    print('bestmove resign')
                if shogi.turn == 1:
                    to_y = 8 - to_y
                    to_x = 8 - to_x
                    from_y = 8 - from_y
                    from_x = 8 - from_x
                    to_y = round(to_y)
                    to_x = round(to_x)
                    from_y = round(from_y)
                    from_x = round(from_x)
                print('bestmove ' + chr(ord('9')-(from_x).astype(int)) + chr(ord('a')+(from_y).astype(int))
                      + chr(ord('9')-(to_x).astype(int)) + chr(ord('a')+(to_y).astype(int)) + ('' if channel < 10 else '+'))
            else:
                if shogi.turn == 1:
                    to_y = 8 - to_y
                    to_x = 8 - to_x
                    to_y = round(to_y)
                    to_x = round(to_x)
                piece = ['P', 'L', 'N', 'S', 'G', 'B', 'R']
                print('bestmove ' + piece[channel-20] + '*'
                      + chr(ord('9')-(to_x).astype(int)) + chr(ord('a')+(to_y).astype(int)))
        elif command[0] == 'quit':
            break
        sys.stdout.flush()


if __name__ == '__main__':
    main()
