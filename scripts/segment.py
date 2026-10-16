#!/usr/bin/env python3
"""Standalone TCP inference server. No ROS imports or ROS Python dependencies."""
import argparse
import socket
import traceback

import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp

from inference_ipc import receive_packet, send_packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5055)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
     
        print('Waiting for image bridge at {}:{}'.format(args.host, args.port), flush=True)
     
        sock, _ = server.accept()
        with sock:
            serve(sock, args)


def serve(sock, args):
    try:
        if not 0 < args.threshold < 1:
            raise ValueError('threshold must be between 0 and 1')

        weights = torch.load(args.weights, map_location='cpu', weights_only=True)
        net = smp.Unet(encoder_name='resnet18', encoder_weights=None, in_channels=3, classes=1).to(args.device)
        
        net.load_state_dict(weights['state_dict'])
        net.eval()
        
        size = weights['size']
        
        mean = np.array([.485, .456, .406], dtype=np.float32)
        std = np.array([.229, .224, .225], dtype=np.float32)
        
        send_packet(sock, {'status': 'ready'})
        
        while True:
            metadata, payload = receive_packet(sock)
            width, height = metadata['width'], metadata['height']
            
            if len(payload) != width * height * 3:
                raise ValueError('Invalid RGB frame size')
            
            image = Image.frombytes('RGB', (width, height), payload)
            resized = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
            
            array = (np.asarray(resized, dtype=np.float32) / 255 - mean) / std
            tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
            
            with torch.inference_mode():
                logits = net(tensor[None].to(args.device))
            
                probability = torch.nn.functional.interpolate(logits.sigmoid(),
                                                              size=(height, width),
                                                              mode='bilinear',
                                                              align_corners=False)[0, 0].cpu().numpy()
            
            mask = (probability >= args.threshold).astype(np.uint8) * 255
            result = mask.tobytes()
           
            if metadata.get('overlay'):
                overlay = Image.blend(image, Image.new('RGB', image.size, (0, 255, 0)), .4)
                result += Image.composite(overlay, image, Image.fromarray(mask)).tobytes()
            
            send_packet(sock, {'status': 'ok',
                               'width': width,
                               'height': height,
                               'overlay': bool(metadata.get('overlay'))},
                                result)
    except EOFError:
        return
    except Exception as exc:
        traceback.print_exc()
        try:
            send_packet(sock, {'status': 'error', 'error': str(exc)})
        except OSError:
            pass
        raise


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
