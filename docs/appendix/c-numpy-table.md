# 부록 C. 수식 ↔ 파이썬 / NumPy 대조표

> 수식을 보고 **"코드로는 뭐지?"** 싶을 때, 또는 그 반대일 때 찾아보세요.

---

## 기본 연산

| 수식 | 파이썬 / NumPy |
|---|---|
| $x^2$ | `x**2` |
| $\sqrt{x}$ | `np.sqrt(x)` |
| $e^x$ | `np.exp(x)` |
| $\log x$ (자연로그) | `np.log(x)` |
| $\log_{10} x$ | `np.log10(x)` |
| $\|x\|$ (절댓값) | `abs(x)` |
| $\max(0, x)$ | `np.maximum(0, x)` |

---

## Σ (Chapter 16)

| 수식 | 코드 |
|---|---|
| $\sum_i x_i$ | `x.sum()` |
| $\sum_i w_i x_i$ | `(w * x).sum()` 또는 `w @ x` |
| $\frac{1}{n}\sum_i x_i$ | `x.mean()` |
| $\sum_i (y_i - \hat{y}_i)^2$ | `((y - yhat)**2).sum()` |
| $\sum_i \sum_j a_{ij}$ | `a.sum()` (전체 합) |

> 💬 **Σ를 for 루프로 짜지 마세요.** NumPy의 `.sum()` 이 수백 배 빠릅니다.
> 원리는 같지만, 내부적으로 최적화되어 있습니다.

---

## 벡터 (Chapter 09~10)

| 수식 | 코드 |
|---|---|
| $\mathbf{a} + \mathbf{b}$ | `a + b` |
| $3\mathbf{a}$ | `3 * a` |
| $\mathbf{a} \cdot \mathbf{b}$ (내적) | `a @ b` 또는 `np.dot(a, b)` |
| $\|\mathbf{a}\|$ (길이) | `np.linalg.norm(a)` |
| 코사인 유사도 | `a @ b / (norm(a) * norm(b))` |

---

## 행렬 (Chapter 11)

| 수식 | 코드 |
|---|---|
| $AB$ (행렬 곱) | `A @ B` |
| $A^T$ (전치) | `A.T` |
| $W\mathbf{x} + \mathbf{b}$ | `W @ x + b` |
| 크기 확인 | `A.shape` |
| $A \odot B$ (같은 자리끼리 곱) | `A * B` |

> ⚠️ **`*` 와 `@` 는 완전히 다릅니다.**
> - `A * B` → 같은 자리끼리 곱 (크기가 같아야 함)
> - `A @ B` → **행렬 곱** (Chapter 11의 그것)
>
> 초보자가 가장 많이 하는 실수입니다.

---

## 통계 (Chapter 15)

| 수식 | 코드 |
|---|---|
| $\bar{x}$ (평균) | `x.mean()` |
| $\sigma$ (표준편차) | `x.std()` |
| 분산 | `x.var()` |
| 중앙값 | `np.median(x)` |
| 표준화 | `(x - x.mean()) / x.std()` |
| 정규화 (0~1) | `(x - x.min()) / (x.max() - x.min())` |
| 상관계수 | `np.corrcoef(x, y)` |

---

## AI 함수들 (Chapter 08)

| 수식 | 직접 구현 | PyTorch |
|---|---|---|
| 시그모이드 | `1 / (1 + np.exp(-x))` | `torch.sigmoid(x)` |
| ReLU | `np.maximum(0, x)` | `torch.relu(x)` |
| 소프트맥스 | `np.exp(x) / np.exp(x).sum()` | `torch.softmax(x, dim=-1)` |
| MSE 손실 | `((y - yhat)**2).mean()` | `nn.MSELoss()` |
| 크로스엔트로피 | `-(y * np.log(yhat)).sum()` | `nn.CrossEntropyLoss()` |

> 💬 실무에서는 직접 구현하지 않습니다. 수치적으로 불안정할 수 있거든요.
> (예: 소프트맥스는 `exp` 가 오버플로 나지 않게 최댓값을 빼고 계산합니다)
>
> 다만 **직접 짜 보면 이해가 확실히 굳습니다.** 한 번쯤 해 보시길 권합니다.

---

## 학습 루프 (Chapter 19)

| 수식 | PyTorch |
|---|---|
| 순전파 $\hat{y} = f(x)$ | `pred = model(x)` |
| 손실 $L$ | `loss = criterion(pred, y)` |
| 기울기 $\nabla L$ | `loss.backward()` |
| 갱신 $\theta \leftarrow \theta - \eta\nabla L$ | `optimizer.step()` |
| 기울기 초기화 | `optimizer.zero_grad()` |

```python
for epoch in range(100):
    pred = model(x)              # 순전파
    loss = criterion(pred, y)    # 손실
    optimizer.zero_grad()        # 기울기 초기화
    loss.backward()              # 역전파
    optimizer.step()             # 갱신
```

> 🎯 **이 다섯 줄이 딥러닝 학습의 전부입니다.** (Chapter 19, 22)

---

## shape 다루기

| 하고 싶은 것 | 코드 |
|---|---|
| 크기 확인 | `x.shape` |
| 모양 바꾸기 | `x.reshape(32, -1)` |
| 차원 추가 | `x[None, :]` 또는 `x.unsqueeze(0)` |
| 차원 제거 | `x.squeeze()` |
| 축 바꾸기 | `x.transpose(0, 1)` |

> 💬 `-1` 은 **"나머지는 알아서 계산해"** 라는 뜻입니다.
> `x.reshape(32, -1)` → 첫 차원은 32로, 나머지는 자동.

---

| ⬅️ [부록 B. 삼각함수](b-trigonometry.md) | 다음 ➡️ [부록 D. 2진법과 부동소수점](d-floating-point.md) |
|---|---|
