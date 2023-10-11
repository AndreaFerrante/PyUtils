import socket
import struct


def send_magic_packet(mac_address):

    """
    Send a Wake-on-LAN magic packet to the specified MAC address.
    """

    # Parse MAC address and convert to bytes...
    mac_address = mac_address.replace(':', '')
    mac_address = bytes.fromhex(mac_address)

    ############################################################
    # Create the magic packet payload...
    magic_packet_payload = b'\xFF' * 6 + mac_address * 16
    ############################################################

    # Send the magic packet...
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(magic_packet_payload, ('255.255.255.255', 9))

