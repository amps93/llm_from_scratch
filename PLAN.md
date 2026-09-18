# 단일 RTX 4060 Ti 16GB에서 PyTorch로 LLM 처음부터 만들기

## 1. 목표와 완료 기준

이 프로젝트의 목적은 좋은 상용 모델을 만드는 것이 아니라, **토크나이저부터 모델 구조, 사전학습, 대화 학습, 평가까지 직접 구현하며 원리를 이해하는 것**이다. 실제 모델과 학습 루프는 PyTorch로 작성하고, 기존 모델의 가중치나 완성된 LLM 아키텍처는 가져오지 않는다. K2 Horizon은 공개 데이터와 설계·학습 절차의 참고 자료로만 사용한다.

완료 기준은 다음과 같다.

- 무작위 초기화한 모델의 다음 토큰 예측 학습이 재현 가능하게 진행된다.
- 분리한 검증 데이터의 loss가 초기값보다 내려가고, 학습·검증 곡선이 기록된다.
- 사전학습 전후와 대화 학습 전후의 한국어 생성 결과를 동일한 질문 세트로 비교할 수 있다.
- 학습 중단 후 체크포인트에서 재개할 수 있고, 설정·데이터 출처·토큰 수·실행 시간이 기록된다.
- 모델의 한계와 실패 사례를 결과 보고서에 남긴다.

참고로 제공된 `LLM_FROM_SCRATCH_GUIDELINE.md`는 **A100 MIG 20GB, 700~800M 파라미터**를 전제로 한다. 여기서는 하드웨어와 사용자의 PyTorch 목표에 맞게 크기와 순서를 다시 정한다. 원리를 더 깊게 확인하고 싶다면 작은 NumPy attention/역전파 구현을 별도 선택 과제로 둘 수 있지만, 본 프로젝트의 필수 단계는 아니다.

## 2. 범위와 기본 선택

| 항목 | 첫 구현 | 이후 실험 |
|---|---|---|
| 프레임워크 | PyTorch로 모델·학습 루프 직접 작성 | 성능 최적화 비교 |
| 모델 | 약 1억 파라미터 dense decoder-only Transformer | 약 3천만 파라미터 축소 실험, 여력이 있으면 2억~3억 파라미터 |
| 초기화 | 전체 가중치 무작위 초기화 | 동일 |
| 문맥 길이 | 1,024 토큰 | 2,048 토큰 |
| 학습 목표 | causal language modeling | 지시 학습, 데이터 혼합 실험 |
| 정밀도 | BF16 autocast, 지원 문제가 있으면 FP16 | FP32 기준 실행과 비교 |
| 데이터 | AI Hub 한국어 사전학습 텍스트 + 선별한 K2 공개 텍스트 | AI Hub 대화·지시 데이터 |

**1억 파라미터는 시작 설계값이지 품질 보증치가 아니다.** 16GB에서 메모리가 맞는지 확인한 다음, 실제 토큰 처리량을 측정해 총 학습량을 확정한다. 처음부터 128K 문맥, MoE, 강화학습, 도구 사용을 구현하지 않는다.

## 3. 모델 설계

첫 기준 모델은 K2의 대형 설정을 복사하지 않고, 이해하기 쉬운 작은 구조로 만든다.

| 구성 요소 | 기준 설정 | 학습 포인트 |
|---|---:|---|
| 어휘 크기 | 32,000 | 한국어·영어 혼합 데이터로 학습한 토크나이저 |
| 층 수 | 12 | 층별 활성값과 연산량 |
| hidden size | 768 | 파라미터 수와 표현력 |
| attention heads / KV heads | 12 / 4 | GQA와 KV 공유 |
| MLP intermediate size | 2,048 | SwiGLU |
| 위치 표현 | RoPE | 상대적 위치 정보 |
| 정규화 | RMSNorm, pre-norm | 학습 안정성 |
| 임베딩 | 입력·출력 가중치 공유 | 파라미터 절약 |
| 최대 학습 문맥 | 1,024 | 활성값 메모리 관리 |

위 설정은 대략 **1억 파라미터**다. 정확한 수치는 구현 뒤 `sum(p.numel() for p in model.parameters())`로 확인한다. 비교를 위해 먼저 일반 multi-head attention을 구현하고, 출력이 맞는지 검증한 뒤 GQA로 확장한다. Attention 계산에는 PyTorch의 `scaled_dot_product_attention`을 활용하되 Q/K/V 투영, RoPE, 마스킹, 헤드 재배열은 직접 작성한다.

모델을 생성하기 전 설정값만으로 예상 파라미터 수를 계산하는 도구를 만든다. 실제 생성 후 센 값과 비교해 embedding 공유 여부, GQA 투영 크기, SwiGLU의 세 행렬이 올바르게 반영됐는지 확인한다. PyTorch의 tensor·autograd·`nn.Linear`·`nn.Embedding`·AdamW·SDPA는 기반 도구로 사용한다. RMSNorm, RoPE, attention/GQA, SwiGLU, block, 생성, 학습 루프는 직접 구성한다.

가이드라인의 **NumPy와 PyTorch 결과 비교** 취지는 작은 PyTorch 기준 구현으로 가져온다. 길이 4~8의 고정 입력과 작은 가중치에서 attention을 행렬곱·mask·softmax로 직접 계산한 결과를 SDPA 결과와 비교한다. 같은 입력에서 loss, 일부 gradient, optimizer 한 번 적용한 후의 가중치도 확인한다. 이 비교를 통과한 뒤 빠른 SDPA 경로로 본 학습을 진행한다.

K2 Horizon의 공개 자료에서 참고할 것은 RoPE·GQA 등 구조적 선택, 데이터 혼합과 정제, 사전학습→중간 학습→사후 학습의 단계 구분, 중간 체크포인트와 평가 기록 방식이다. 공개 0.9B 설정(28층, hidden 1,536)을 그대로 축소 복제하면 이 GPU의 학습 시간 목표에 맞지 않을 수 있다. MoVA·MoE는 기본 모델이 완성된 뒤 별도 연구 주제로 남긴다.

## 4. 데이터 계획

### 4.1 사전학습용

1. [AI Hub 사전 학습 데이터](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=wblextrldata&currMenu=520&dataSetSn=71898&topMenu=100)의 한국어 일반·도메인·합성 텍스트를 우선 검토한다. 페이지의 총량 약 1.09T 토큰은 한국어만의 양이 아니므로 항목별로 선별한다.
2. [K2 Horizon 데이터 모음](https://huggingface.co/collections/IFM/k2-horizon)에서 [TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) 같은 텍스트 하위 집합을 소량 표본 추출한다. 전체 저장소는 TB 단위이므로 필요한 shard 또는 streaming만 사용한다.
3. 데이터 원문을 `text` 필드 중심의 공통 형식으로 변환한다. 출처, 원본 식별자, 언어, 해시를 함께 기록한다.
4. 빈 문서·너무 짧은 문서·깨진 인코딩·명백한 중복·개인정보 패턴을 걸러낸다. 표본을 눈으로 확인하고 정제 전후 건수를 기록한다.
5. **문서 단위**로 train/validation을 분리한 뒤 토크나이즈한다. 거의 같은 문서가 양쪽에 들어가지 않도록 중복 검사한다.
6. 본 학습에서는 매 step 원문을 토큰화하지 않는다. 정제된 문서를 미리 token IDs로 바꾸고 EOS를 추가한 뒤, 시퀀스 packing과 shard 저장을 수행한다. 원문 출처와 토큰 shard를 연결할 수 있게 manifest를 남긴다.

packing 실험에서는 **문서 경계를 넘는 attention을 허용할지** 먼저 정한다. 단순 연결 방식은 EOS 뒤의 새 문서를 앞 문서에서 볼 수 있으므로 구현은 쉽지만 문서 간 정보가 섞인다. 초기 버전은 EOS 연결 방식으로 시작하고, 추후 문서 경계 attention mask를 적용한 결과와 비교한다.

첫 데이터 목표는 **정제된 1억~3억 토큰**이다. 이는 고정 요구사항이 아니라 성능 측정 전의 실험 범위다. 초기 비교용으로 한국어 중심 혼합을 사용하고, K2 영어 데이터의 혼합 비율은 별도 실험 변수로 둔다. 토크나이저는 train 자료의 대표 표본으로만 학습하고 validation은 토크나이저 학습에서도 제외한다.

토크나이저 후보는 SentencePiece Unigram/BPE 또는 Hugging Face Tokenizers BPE다. 직접 작성할 부분은 학습용 표본 선정, 특수 토큰 설계, 학습·평가 절차이며 토크나이저 알고리즘 자체를 새로 구현할 필요는 없다. 한국어·영어·숫자·코드·한영 혼합 문장에 대해 문자당 토큰 수와 문장당 토큰 수를 비교한다. 어휘 크기는 16k/32k 후보를 작은 표본에서 비교한 뒤 확정한다.

### 4.2 대화 학습용

[AI Hub Synthetic & Instruction 데이터](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=wblextrldata&currMenu=520&dataSetSn=71903&topMenu=100)와 [고품질 멀티턴 데이터](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=wblextrldata&currMenu=520&dataSetSn=71902&topMenu=100)를 검토한다. 사전학습을 완료한 **자체 모델의 가중치**에서 이어서 학습한다. `user`/`assistant` 턴을 일관된 채팅 형식으로 바꾸고, 기본 실험에서는 assistant 응답 토큰에만 loss를 준다. 합성 데이터는 표본의 한국어 자연스러움, 답의 정확성, 중복을 검사한 뒤 사용한다.

AI Hub 데이터 다운로드에는 이용 신청·승인 절차가 있을 수 있다. 데이터별 이용 조건을 확인하고 원본 파일을 프로젝트 공개 저장소에 올리지 않는다. K2 데이터도 하위 저장소마다 라이선스가 다르므로 출처와 조건을 따로 기록한다.

## 5. 구현 순서와 통과 기준

| 단계 | 구현·실험 | 다음 단계로 넘어가는 기준 |
|---|---|---|
| 0. 환경 측정 | GPU, CUDA, PyTorch 버전·BF16 지원·디스크 여유 확인 | 소형 행렬 연산과 저장/재개 성공 |
| 1. 최소 모델 | 토큰 임베딩, causal attention, MLP, norm, logits, loss | 작은 배치를 과적합시켜 loss가 크게 감소; 간단한 기준 계산과 attention·loss·gradient 비교 |
| 2. 토크나이저·데이터 | 토크나이저 학습, 문서 정제, 토큰화, 고정 길이 시퀀스 구성 | 인코드/디코드 확인, train/validation 누출 검사 |
| 3. 학습 루프 | AdamW, 학습률 warmup/감쇠, gradient clipping, gradient accumulation, autocast, 체크포인트 | 중단·재개 결과와 loss 기록이 일관됨 |
| 4. 사전학습 예비 실행 | 약 1천만~3천만 파라미터 모델로 짧은 학습, 이후 1억 모델 1천~1만 step 측정 | OOM 없이 안정적, 처리량·메모리·예상 시간이 기록됨 |
| 5. 본 사전학습 | 확정한 데이터량으로 학습, 주기적 검증·샘플 생성 | 검증 loss와 한국어 생성의 변화 확인 |
| 6. 대화 학습 | 선별한 AI Hub 대화 데이터로 SFT | 동일 질문 세트에서 지시 수행 변화 확인 |
| 7. 선택 확장 | 문맥 길이 확장용 continued pretraining, 양자화, 로컬 추론 | 본 실험 완료 후 각각 독립적으로 비교 |
| 8. 정리 | 결과·실패 사례·설정·비용 기록 | 다른 사람이 같은 데이터 접근 조건에서 실험 재현 가능 |

예상 디렉터리 구조: `src/model/`, `src/data/`, `src/train/`, `src/eval/`, `configs/`, `scripts/`, `reports/`. 모델 설정과 학습 하이퍼파라미터는 파일로 관리하고 코드에 고정하지 않는다. 원본 데이터와 체크포인트는 Git 추적에서 제외하고 경로만 설정 파일에 기록한다. **코드를 작성할 때 모든 함수에는 기능과 사용법을 알 수 있는 한글 설명을 붙인다.** 환경 비밀 파일인 `.env`는 직접 열어보지 않는다.

초기 optimizer 기준안은 가이드라인을 따라 AdamW `betas=(0.9, 0.95)`, `weight_decay=0.1`, warmup 뒤 cosine decay로 둔다. 학습률·warmup 길이·유효 배치 크기는 예비 실행에서 결정하며 설정 파일에 남긴다. 기준안이 항상 최적이라는 뜻은 아니다.

## 6. 메모리·시간 관리

- BF16 파라미터만 놓고 보면 1억 개는 약 0.2GB지만, 학습에는 FP32 optimizer 상태·gradient·활성값·임시 버퍼가 추가된다. 따라서 파라미터 크기만으로 적합성을 판단하지 않는다.
- 첫 시도는 micro-batch 1~2, 길이 1,024, gradient accumulation으로 유효 배치 크기를 맞춘다. 메모리가 부족하면 길이·micro-batch를 먼저 낮추고 gradient checkpointing을 검토한다.
- 100~500 step의 예비 실행에서 `tokens/sec`, 최대 VRAM, step 시간, 검증 loss를 측정한다. **총 예상 시간 = 목표 토큰 수 ÷ 실측 tokens/sec**로 계산하고 목표 토큰 수를 조정한다.
- 데이터 준비 시간과 체크포인트 저장 시간을 별도로 기록한다. 긴 학습은 일정 step마다 모델·optimizer·scheduler·난수 상태, global step, 누적 학습 토큰 수, 설정을 저장한다. 재개 시 같은 데이터 순서로 이어지는지도 확인한다.
- train/validation loss, perplexity, learning rate, gradient norm, tokens/sec, step 시간, 최대 VRAM, 누적 학습 토큰 수를 최소 로그로 남긴다.

## 7. 평가 설계

고정된 질문 세트에 한국어 설명, 짧은 요약, 간단한 추론, 영어 문장, 문맥 밖 사실 질문을 넣는다. 사전학습 전/후와 대화 학습 후에 같은 생성 설정으로 비교한다. 정량 지표는 **train/validation loss와 perplexity**를 기본으로 사용한다. 생성 품질은 정량 지표와 분리해 사람이 오류 유형을 적는다. 작은 스크래치 모델이 사실을 모르는 것은 정상적인 결과이며, 근거 없는 답변과 질문 형식 이해 실패를 따로 센다.

추가 실험은 한 번에 변수 하나만 바꾼다: GQA 유무, 한국어/영어 혼합 비율, 문맥 길이, 데이터 정제 수준, 모델 크기. 각 실험의 seed, 코드 버전, 데이터 버전, 총 학습 토큰 수, GPU 시간, 결과를 같은 표에 기록한다.

가이드라인의 **토큰 수별 체크포인트 평가**도 적용한다. 예를 들어 1천만·5천만·1억·2억 학습 토큰 시점에서 같은 검증 세트와 생성 질문을 실행하고, 다음 질문에 답할 수 있게 짧은 실험 노트를 남긴다: 토크나이저 변경이 한국어 토큰 수를 얼마나 줄였는가? GQA와 MHA의 출력·메모리는 어떻게 달랐는가? 문맥 길이와 micro-batch가 VRAM에 미친 영향은 얼마인가? 학습 토큰이 늘 때 검증 loss와 생성 결과가 함께 개선됐는가?

## 8. 참고 자료

- [K2 Horizon 소개와 공개 범위](https://ifm.ai/blog/k2/)
- [K2 Horizon 0.9B 기술 부록](https://huggingface.co/IFM/K2-Horizon-0.9B/blob/main/APPENDIX.md)
- [K2 Horizon 공개 데이터 모음](https://huggingface.co/collections/IFM/k2-horizon)
- [AI Hub 독자 AI모델 데이터 목록](https://aihub.or.kr/aihubdata/wblextrldata/list.do?currMenu=520&topMenu=100)
- [PyTorch scaled dot product attention 문서](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
