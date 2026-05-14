import re
import socket


def send_magic_packet(mac_address: str) -> None:
    """Send a Wake-on-LAN magic packet to the given MAC address.

    Accepts colon- or hyphen-separated MAC addresses (e.g. AA:BB:CC:DD:EE:FF).
    Prints a success message on delivery; prints an error message on failure.
    """
    try:
        if not re.match(r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$", mac_address):
            raise ValueError(f"Invalid MAC address format: {mac_address}")

        mac_bytes = bytes.fromhex(mac_address.replace(":", "").replace("-", ""))
        magic_packet = b"\xff" * 6 + mac_bytes * 16

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(magic_packet, ("255.255.255.255", 7))

        print("Magic packet sent successfully.")
    except Exception as ex:
        print(f"Failed to send magic packet: {ex}")
