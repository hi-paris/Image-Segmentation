## 3️⃣ Create different configurations 

### i) Normal 008
This configuration is based on the normal training configuration on which we removed the Is Thing/ Out Vocab classes labels. 
The following command generates the json file needed for the training.

```python
python create_configs.py Normal
```
### ii) Invocab
This configuration is obtained by applying FCCLIP inference where we replace the In vocab/ Stuff by ground truth.
The following command generates the json file needed for the training.

```python
python create_configs.py Invocab
```
### iii) Naive 
This configuration is the intersection of the Invocab approach and the normal training configuration.
The following command generates the json file needed for the training.

```python
python create_configs.py Naive
```