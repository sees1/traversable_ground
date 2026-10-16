"""Framed JSON metadata and raw bytes; compatible with Python 3.8 and 3.11."""
import json
import struct

MAX_METADATA = 65536
MAX_PAYLOAD = 256 * 1024 * 1024


def read_exact(sock, size):
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(min(size - len(result), 1024 * 1024))
        if not chunk:
            raise EOFError('Inference connection closed')
        result.extend(chunk)
    return bytes(result)


def send_packet(sock, metadata, payload=b''):
    header = json.dumps(metadata).encode('utf-8')
    if len(header) > MAX_METADATA or len(payload) > MAX_PAYLOAD:
        raise ValueError('Inference packet too large')
    sock.sendall(struct.pack('!IQ', len(header), len(payload)))
    sock.sendall(header)
    sock.sendall(payload)


def receive_packet(sock):
    header_size, payload_size = struct.unpack('!IQ', read_exact(sock, 12))
    if header_size > MAX_METADATA or payload_size > MAX_PAYLOAD:
        raise ValueError('Inference packet too large')
    metadata = json.loads(read_exact(sock, header_size).decode('utf-8'))
    return metadata, read_exact(sock, payload_size)
