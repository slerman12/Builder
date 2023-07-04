# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import getpass
import os

from cryptography.fernet import Fernet

from tributaries import __file__


def get_pass(kind='pass', reenter=False):
    # Get password, encrypt, and save for reuse -- locally only of course
    if os.path.exists(f'{__file__.rsplit("/", 1)[0]}/{kind}') and not reenter:
        with open(f'{__file__.rsplit("/", 1)[0]}/{kind}', 'r') as file:
            key, encoded = file.readlines()
            password = Fernet(key).decrypt(bytes(encoded, 'utf-8'))
        password = password.decode()
    else:
        password, key = getpass.getpass(f'Enter {kind} pass:'), Fernet.generate_key()
        encoded = Fernet(key).encrypt(bytes(password, 'utf-8'))
        with open(f'{__file__.rsplit("/", 1)[0]}/{kind}', 'w') as file:
            file.writelines([key.decode('utf-8') + '\n', encoded.decode('utf-8')])
    return password
