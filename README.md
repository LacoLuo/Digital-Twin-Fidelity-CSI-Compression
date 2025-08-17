# Digital Twin Aided Massive MIMO CSI Feedback: Exploring the Impact of Twinning Fidelity
This is the code package related to the following article: "[Digital Twin Aided Massive MIMO CSI Feedback: Exploring the Impact of Twinning Fidelity]()"

# To-do list
Once the paper is publicly available, we will release the scripts for the following items:
- [ ] Python scripts for generating 3D geometry model of different fidelity levels
- [ ] Fine-tuning scripts
- [ ] Scripts for plotting the figures of main results

# Code package content
**Download the ray-tracing results**
- Download the ray-tracing results via this [Dropbox link](https://www.dropbox.com/scl/fo/647vt25xzw0r7u966bmgq/ACYnCeYvy9f9zkrBMqriC_M?rlkey=zi3f7pdp1a74ac9bmzmebw332&st=kp0f2iaa&dl=0).
- Move `Raytracing_scenarios/` to the `DeepMIMO/` directory.

**Generate datasets using DeepMIMO**
- Set up the scenario in `parameters.m`.
- Run `DeepMIMO_Dataset_Generator.m` to generate channel data.
- Run `process_raw_data.m` to pre-process the data.

**Train the model**
1. Generate training and testing datasets
```
python gen_csv.py
```
2. Set up the training parameters in `config.py`
3. Run the training sessions
```
python train.py
```
