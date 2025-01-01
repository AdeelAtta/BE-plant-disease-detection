import torch
import torch.nn as nn
import json

class CNN_NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        
        self.res1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        
        self.res2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        )
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 38)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def test_forward_pass(model, input_size):
    """Test forward pass with given input size"""
    x = torch.randn(1, 3, input_size, input_size)
    try:
        print(f"\nTesting forward pass with input size: {input_size}x{input_size}")
        out = model(x)
        print("Forward pass successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        return True
    except Exception as e:
        print(f"Error with size {input_size}x{input_size}: {str(e)}")
        return False

def convert_to_onnx():
    try:
        print("Loading PyTorch model...")
        model = CNN_NeuralNet()
        state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()

        # Test different input sizes
        input_sizes = [256, 384, 512]
        successful_size = None
        
        for size in input_sizes:
            if test_forward_pass(model, size):
                successful_size = size
                break
        
        if successful_size is None:
            raise Exception("Could not find a working input size")
            
        print(f"\nUsing input size: {successful_size}x{successful_size}")
        dummy_input = torch.randn(1, 3, successful_size, successful_size)
        
        print("Converting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            'model.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print("Model converted successfully!")
        
        # Verify the model
        import onnx
        onnx_model = onnx.load('model.onnx')
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
        
        # Save the input size for later use
        config = {
            'input_size': successful_size,
            'num_classes': 38
        }
        with open('model_config.json', 'w') as f:
            json.dump(config, f)
        print(f"Model configuration saved to model_config.json")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    convert_to_onnx()