#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Ensure Homebrew is installed
if command_exists brew; then
    echo "Homebrew is already installed."
else
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install CMake if not present or if it's below version 3.24
if command_exists cmake; then
    CMAKE_VERSION=$(cmake --version | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
    if [[ $(echo "$CMAKE_VERSION < 3.24" | bc -l) -eq 1 ]]; then
        echo "CMake version is below 3.24. Installing the latest version..."
        brew upgrade cmake
    else
        echo "CMake version $CMAKE_VERSION is already installed."
    fi
else
    echo "Installing CMake..."
    brew install cmake
fi

# Install Git if not present
if command_exists git; then
    echo "Git is already installed."
else
    echo "Installing Git..."
    brew install git
fi

# Install Git-LFS if not present
if command_exists git-lfs; then
    echo "Git LFS is already installed."
else
    echo "Installing Git LFS..."
    brew install git-lfs
    git lfs install
fi

# Install Rust and Cargo if not present
if command_exists cargo; then
    echo "Rust and Cargo are already installed."
else
    echo "Installing Rust and Cargo..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    . "$HOME/.cargo/env"  # Ensure Rust is available in this shell
fi

# Install mlc-llm-nightly-cpu and mlc-ai-nightly-cpu using pip
if command_exists pip; then
    echo "Installing mlc-llm-nightly-cpu and mlc-ai-nightly-cpu..."
    pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
else
    echo "pip is not installed. Please install Python and pip."
    exit 1
fi

# Ensure Xcode is installed
if command_exists xcode-select; then
    echo "Xcode is already installed."
else
    echo "Installing Xcode..."
    xcode-select --install
fi

# Accept Xcode license if not accepted
echo "Accepting Xcode license..."
sudo xcodebuild -license accept

# Ensure Xcode developer tools are correctly set
echo "Setting Xcode developer directory..."
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# Ensure iOS SDK is installed
echo "Checking iOS SDK..."
xcrun --sdk iphoneos --show-sdk-path
if [ $? -ne 0 ]; then
    echo "iOS SDK is not installed. Please ensure that Xcode is fully installed with all necessary SDKs."
    exit 1
else
    echo "iOS SDK is installed."
fi

# Add Xcode developer tools to PATH
echo "Adding Xcode developer tools to PATH..."
export PATH="/Applications/Xcode.app/Contents/Developer/usr/bin:$PATH"

echo "All dependencies are installed."

# Run the Python script
echo "Running the Python script..."
python ios_app.py
