# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from pexpect import spawn

from tributaries.SafePass import get_pass


def connect_vpn(username='slerman'):
    def _connect_vpn():
        try:
            print('Connecting...')
            password = get_pass('bluehive')
            p = spawn('/opt/cisco/anyconnect/bin/vpn connect vpnconnect.rochester.edu')
            p.expect('Username: ')
            p.sendline('')
            p.expect('Password: ')
            p.sendline(password)
            p.expect('Second Password: ')
            print('A Duo two-factor authentication push notification may have been sent.')
            p.sendline('push')
            p.expect('VPN>')
        except Exception:
            pass
        print(f'Connected to VPN\n'
              f'If needed, log into Bluehive manually with:\n'
              f'\tssh {username}@bluehive.circ.rochester.edu\n'
              f'and approve the Duo two-factor authentication push notification that\'s sent.\n'
              f'If issues, try restarting Cisco AnyConnect,\n'
              f'or read the instructions for installing the Cisco AnyConnect client '
              f'for connecting to the University of Rochester VPN:\n'
              f'https://tech.rochester.edu/services/remote-access-vpn/')
    return _connect_vpn


def disconnect_vpn():
    p = spawn('/opt/cisco/anyconnect/bin/vpn disconnect')
    p.expect('b')


if __name__ == '__main__':
    connect_vpn()()
