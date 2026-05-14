
def send_magic_packet(mac_address: str) -> None:

    import socket

    try:
        import re
        if not re.match(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$', mac_address):
            raise ValueError(f"Invalid MAC address format: {mac_address}")
        mac_address = mac_address.replace(':', '').replace('-', '')
        mac_address_bytes = bytes.fromhex(mac_address)
        print(f"MAC Address Bytes: {mac_address_bytes}")

        magic_packet_payload = b'\xFF' * 6 + mac_address_bytes * 16
        print(f"Magic Packet Payload: {magic_packet_payload}")

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            bytes_sent = sock.sendto(magic_packet_payload, ('255.255.255.255', 7))
            print(f"Bytes Sent: {bytes_sent}")

        print("Magic packet sent successfully.")

    except Exception as e:
        print(f"Failed to send magic packet: {e}")
