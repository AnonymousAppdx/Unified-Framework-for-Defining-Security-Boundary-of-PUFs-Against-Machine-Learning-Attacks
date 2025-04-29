## Unified-Framework-for-Defining-Security-Boundary-of-PUFs-Against-Machine-Learning-Attacks
Physical Unclonable Functions (PUFs) serve as lightweight, hardware-intrinsic entropy sources widely deployed in IoT security applications. However, delay-based PUFs have been shown to be vulnerable to Machine Learning Attacks (MLAs), undermining their assumed unclonability. Existing evaluations of PUF security predominantly rely on empirical modeling experiments, which lack theoretical guarantees and are highly sensitive to advances in machine learning techniques. This reliance raises critical concerns about the soundness and comparability of PUF security across architectures and adversarial models.

In this work, we propose a novel, formal, and unified probabilistic framework for evaluating PUF security against MLAs, independent of specific attack models or learning algorithms. We mathematically characterize the adversary’s advantage in predicting responses to unseen challenges based solely on observed challenge-response pairs (CRPs), formulating the problem as a conditional probability estimation over the space of candidate PUFs. Our framework enables principled, interpretable, and architecture-agnostic security evaluation without requiring any machine learning model training or empirical calibration. 

Through extensive analysis of Arbiter PUFs, XOR PUFs, Feed-Forward PUFs, and several obfuscation-based constructions, we demonstrate that the proposed approach systematically quantifies PUF resilience, captures subtle security differences, and provides actionable, theoretically grounded security guarantees for practical PUF deployment.

## Codes
1. You can start your evaluation using
```bash
sh mtkl.sh
```
2. To modify the evalutaion logics, you can find in mtkl.py
```python
def compute_PUF_response(Q_list, M, model="xor-1", ff_params=None):
    ...
```
3. You can reproduce all the figures in the paper by modify the folder name you just generated from mtkl.sh, and then:
```bash
python plot.csv.py
```
