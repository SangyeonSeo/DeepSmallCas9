1. System Requirements:
	Ubuntu	16.04
	Python	2.7.12
	Python Packages:
		numpy 1.14.5
		scipy 1.1.0

	Tensorflow and dependencies:
		Tensorflow  1.4.1
		CUDA	    8.0.61
		cuDNN	    5.1.10

2. Installation Guide (required time, <120 minutes):

- Operation System
	Ubuntu 16.04 download from https://www.ubuntu.com/download/desktop
	
- Python and packages
	Download Python 2.7.12 tarball on https://www.python.org/downloads/release/python-2712/
	Unzip and install:
		tar -zxvf Python-2.7.12.tgz
		cd ./Python-2.7.12
		./configure
		make

	Package Installation:
		pip install numpy==1.14.5
		pip install scipy==1.1.0

	Tensorflow Installation:
		(for GPU use)
		pip install tensorflow-gpu==1.4.1
		(for CPU only)
		pip install tensorflow==1.4.1


(for GPU use)

- CUDA Toolkit 8.0 
	wget -O cuda_8_linux.run https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
	sudo chmod +x cuda_8_linux.run
	./cuda_8.0.61_375.26_linux.run

- cuDNN 5.1.10
	Download CUDNN tarball on https://developer.nvidia.com/cudnn
	Unzip and install:
		tar -zxvf cudnn-8.0-linux-x64-v5.1.tgz 

For more details, please refer to CUDA, CuDNN, and Tensorflow installation guide on Github: 			
	https://gist.github.com/ksopyla/813a62d6afc4307755e5832a3b62f432


3. Demo Instructions (required time, <1 min):

Input1: ./dataset/        # List of Target Sequence(s)
	File format:
	sRGN3.1 (4 bp + 21 bp protospacer + 4 bp PAM + 3 bp)
	SlugCas9 (4 bp + 21 bp protospacer + 4 bp PAM + 3 bp)
	SaCas9 (4 bp + 21 bp protospacer + 6 bp PAM + 3 bp)
	SauriCas9 (4 bp + 21 bp protospacer + 4 bp PAM + 3 bp)
	Sa-SlugCas9 (4 bp + 21 bp protospacer + 4 bp PAM + 3 bp)
	SaCas9-KKH (4 bp + 21 bp protospacer + 6 bp PAM + 3 bp)
	eSaCas9 (4 bp + 21 bp protospacer + 6 bp PAM + 3 bp)
	efSaCas9 (4 bp + 21 bp protospacer + 6 bp PAM + 3 bp)
	SauriCas9-KKH (4 bp + 21 bp protospacer + 4 bp PAM + 3 bp)
	SlugCas9-HF (4 bp + 21 bp protospacer + 4 bp PAM + 3 bp)
	SaCas9-HF (4 bp + 21 bp protospacer + 6 bp PAM + 3 bp)
	SaCas9-KKH-HF (4 bp + 21 bp protospacer + 6 bp PAM + 3 bp)
	St1Cas9 (4 bp + 19 bp protospacer + 6 bp PAM + 3 bp)
	Nm1Cas9 (4 bp + 23 bp protospacer + 8 bp PAM + 3 bp)
	enCjCas9 (4 bp + 22 bp protospacer + 8 bp PAM + 3 bp)
	CjCas9 (4 bp + 22 bp protospacer + 8 bp PAM + 3 bp)
	Nm2Cas9 (4 bp + X + 22 bp protospacer + 7 bp PAM + 3 bp for matched target) or (4 bp + 23 bp protospacer + 7 bp PAM + 3 bp for mismatched target)

Input2: ./CAS9_TYPE/ # Pre-trained Weight Files
CAS9_TYPE : Cj, efSa, enCj, eSa, Nm1, Nm2, Sa, Sa-HF, Sa-KKH, Sa-KKH-HF, Sa-Slug, Sauri, Sauri-KKH, Slug, Slug-HF, sRGN, St1

Output: ./CAS9_TYPE+"_Test"/"TEST_OUTPUT.xlsx"  

Run script:
	python ./MainTest.py --mode=CAS9_TYPE --filename=TESTFILENAME # replace CAS9_TYPE and TESTFILENAME for your usage.