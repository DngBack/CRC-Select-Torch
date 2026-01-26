#!/usr/bin/env python3
"""
Quick Demo Script for SelectiveNet
This demonstrates the SelectiveNet model on a small CIFAR-10 example
"""
import sys
import os
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), './')
sys.path.append(base)

import torch
import torchvision
import torchvision.transforms as transforms
from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet

def main():
    print("=" * 60)
    print("SelectiveNet Quick Demo")
    print("=" * 60)
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load a small sample of CIFAR-10
    print("\n✓ Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Take only first 10 samples for quick demo
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=10, 
        shuffle=False
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create model
    print("✓ Creating SelectiveNet model...")
    features = vgg16_variant(32, 0.3).to(device)
    model = SelectiveNet(features, 512, 10).to(device)
    model.eval()
    
    # Run inference on sample batch
    print("\n✓ Running inference on 10 test images...")
    print("-" * 60)
    
    with torch.no_grad():
        images, labels = next(iter(testloader))
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        out_class, out_select, _ = model(images)
        
        # Get predictions
        _, predicted = torch.max(out_class.data, 1)
        selection_scores = torch.sigmoid(out_select).squeeze()
        
        print(f"{'Image':<8} {'True Label':<15} {'Predicted':<15} {'Confidence':<12} {'Select?':<10}")
        print("-" * 60)
        
        for i in range(10):
            true_label = classes[labels[i]]
            pred_label = classes[predicted[i]]
            confidence = torch.softmax(out_class[i], dim=0).max().item()
            select_score = selection_scores[i].item()
            will_select = "✓ Accept" if select_score > 0.5 else "✗ Reject"
            
            print(f"{i+1:<8} {true_label:<15} {pred_label:<15} {confidence:>6.2%}       {will_select:<10}")
    
    print("-" * 60)
    print("\n✓ Demo completed!")
    print("\nℹ️  Note: This is an untrained model, so predictions are random.")
    print("   To see real performance, train the model first using:")
    print("   cd scripts && python3 train.py --dataset cifar10 --coverage 0.8 --num_epochs 50 --unobserve")
    print("=" * 60)

if __name__ == '__main__':
    main()

