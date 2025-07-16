# 🔱 Triton Kernel Performance Agent (TKPerfAgent)
TKPerfAgent takes in a user input Triton kernel, rewrites and generates an optimized Triton kernel.
## 🌟 Key Features
- **Automatic Performance Optimization**: TKPerfAgent automatically optimizes the performance of the input Triton kernel simply enabled by a decorator.
- **Extenable Optimization Database**: The agent is empowered by an optimization database, which can be extended to support more optimization techniques.
- **Iterative Error Correction**: The agent iterates by learning from previous compilation errors and ends up by generating a correct Triton kernel or timeout.

## 💻 Quick Start
1. Fork and clone the repository:
```
git clone <your-private-repo-url> TKPerfAgent
cd TKPerfAgent
```
2. Create a virtual environment:
```
conda create -n <env-name> python=3.13
conda activate <env-name>
```
3. Install the required dependencies:
```
pip install -e .
```
4. Use personal tokens:
```
cp .env.example .env
```
Fill in personal tokens in .env file.

5. Run the example tests:
```
cd examples
python gemm.py
```

## 🕹️ Configurations
```
export DEBUG_MODE=1 # 1 = enable debug mode; 0 = disable debug mode
export NUM_OF_ROUNDS=10 # set the max number of rounds for LLMs to correct compilation errors
export MAX_TOKENS=4096 # set the max number of tokens for LLMs
```

## 🛠️ Usage
### ⌨️ Command Line Interface
To enable the agent, the user must add the following decorator to the top of their Triton kernel:
```
@kernel_opt(
    func_prompt = <functional-description>,   # str
    opt_prompt = <optimization-hint>,         # str
    model = <llm-model-name>,                 # str
    dsl = <dsl-level>,                        # str
    kernel_name = <kernel-name>,              # str
    debug = <enable-debug>,                   # bool
)
```
An example of a use case is:
```
@kernel_opt(
    func_prompt="an unoptimized matrix-matrix multiplication",
    opt_prompt="Integrate persistent programming style",
    model="gpt-3.5-turbo",
    dsl="Triton",
    kernel_name="gemm",
    debug=True,
)
```
### 🛜 Web Interface
The agent can also be used through a web interface. To start the web server, run:

```
streamlit run web_ui.py
```

The default port number is 8501.
You can access the web interface by visiting http://localhost:\<port-number\> in your browser.  

The web UI looks like this:  
<p align="center">
<img width="612" height="442" alt="Image" src="https://github.com/user-attachments/assets/e8b76510-e1eb-4b58-acee-3d72d65d8916" />
<p/>

## 🔮 Next Steps
- [ ] Automatic knowledge database construction
- [ ] End to end application testing
