# Vibration Diagnostics

This project utilizes event-frame filtering and a genetic algorithm to detect and measure vibrations of an object using high-speed camera footage. The integration of neuromorphic computing and high-speed cameras aims to revolutionize vibration diagnostics by making advanced vibration analysis techniques accessible to everyone.

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgments](#acknowledgments)

## Overview

Vibration analysis is crucial in various fields such as structural health monitoring, mechanical engineering, and industrial maintenance. Traditional methods often require expensive equipment and specialized expertise. This project seeks to overcome these barriers by leveraging cutting-edge technologies and computer vision algorithms.

## How It Works

### Key Features

1. **Data Retrieval and Preprocessing**
   - Retrieve data from transformed frames of a high-speed camera .
   - Enhance the final results by clearing, normalizing, and denoising the obtained frames.

2. **Frame Correlation and Template Matching**
   - Find the region of the source frame that matches the given image template using the normalized mutual correlation function.
   - Estimate the degree of correlation without scaling or rotating the template.

3. **Visualization**
   - Develop a user-friendly application for interaction and result visualization.

### Algorithm Steps

1. **Event-Frame Filtering**
   - Utilize high-speed cameras and different image enhancing algorithms to capture dynamic scenes and filter relevant events for vibration analysis.

2. **Genetic Algorithm**
   - Apply a genetic algorithm to optimize the detection and measurement of vibrations from the processed frames.

3. **Vibration Measurement**
   - Analyze the processed frames to detect and measure the vibrations accurately.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/argirovga/vibration_diagnostics.git
cd vibration_diagnostics
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. **Preprocess Data**

   Use the provided scripts to preprocess the data from the high-speed or event-based cameras.

   ```bash
   python run.py
   ```

### Jupyter Notebook

You can also explore the provided Jupyter Notebook for a step-by-step walkthrough of the frame shifting and vibration calculation.

```bash
jupyter notebook frame_shifting_and_vibration_calculation.ipynb
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to all contributors and the open-source community.
- This project is inspired by the need for accessible and advanced vibration diagnostic tools.
