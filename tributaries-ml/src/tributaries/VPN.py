# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from pexpect import spawn

from tributaries.SafePass import get_pass


def connect_vpn(username='slerman'):
    def _connect_vpn():
        try:
            password = get_pass('bluehive')
            p = spawn('/opt/cisco/anyconnect/bin/vpn connect vpnconnect.rochester.edu')
            p.expect('Username: ')
            p.sendline('')
            p.expect('Password: ')
            p.sendline(password)
            p.expect('Second Password: ')
            p.sendline('push')
            p.expect('VPN>')
        except Exception:
            pass
        print(f'Connected to VPN\n'
              f'For Bluehive: ssh {username}@bluehive.circ.rochester.edu\n'
              f'If issues, try restarting Cisco AnyConnect.\n'
              f'Or read the instructions for installing the Cisco AnyConnect client:\n'
              f'https://tech.rochester.edu/services/remote-access-vpn/')
    return _connect_vpn


def disconnect_vpn():
    p = spawn('/opt/cisco/anyconnect/bin/vpn disconnect')
    p.expect('b')


if __name__ == '__main__':
    connect_vpn()()
