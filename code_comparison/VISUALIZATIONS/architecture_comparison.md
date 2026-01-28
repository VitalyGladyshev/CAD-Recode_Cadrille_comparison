# Архитектурные диаграммы моделей CAD-Recode и Cadrille

## Архитектура CAD-Recode

```mermaid
graph TB
    subgraph Input[Входные данные]
        PC[Point Cloud<br/>256 точек x 3 координаты]
    end
    
    subgraph Encoder[Кодирование]
        FPE[FourierPointEncoder<br/>Fourier encoding<br/>51 dim -> hidden_size]
    end
    
    subgraph BaseModel[Базовая модель Qwen2]
        Embed[embed_tokens<br/>Текстовые embeddings]
        Replace[Замена pad токенов<br/>attention_mask == -1]
        Transformer[Qwen2Model<br/>Transformer layers]
        LMHead[lm_head<br/>Генерация токенов]
    end
    
    subgraph Output[Выход]
        Logits[Logits<br/>vocab_size]
        Code[CadQuery код]
    end
    
    PC --> FPE
    FPE --> Replace
    Input --> Embed
    Embed --> Replace
    Replace --> Transformer
    Transformer --> LMHead
    LMHead --> Logits
    Logits --> Code
    
    style PC fill:#e1f5ff
    style FPE fill:#fff4e1
    style Transformer fill:#e8f5e9
    style Code fill:#fce4ec
```

## Архитектура Cadrille

```mermaid
graph TB
    subgraph Input[Входные данные]
        PC[Point Cloud<br/>256 точек]
        IMG[Изображения<br/>PIL.Image]
        VID[Видео<br/>Множественные изображения]
        TEXT[Текст<br/>Описание]
    end
    
    subgraph Encoding[Кодирование]
        FPE[FourierPointEncoder<br/>Fourier encoding]
        VE[Visual Encoder<br/>Qwen2VL visual]
        TE[Text Encoder<br/>embed_tokens]
    end
    
    subgraph Integration[Интеграция]
        ImgEmbed[Встраивание изображений<br/>masked_scatter image_token_id]
        VidEmbed[Встраивание видео<br/>masked_scatter video_token_id]
        PCEmbed[Встраивание PC<br/>start_idxs цикл]
        Merge[Объединение embeddings]
    end
    
    subgraph BaseModel[Базовая модель Qwen2VL]
        RoPE[RoPE с визуальными токенами<br/>get_rope_index]
        Transformer[Qwen2VLModel<br/>Transformer layers]
        LMHead[lm_head<br/>Генерация токенов]
    end
    
    subgraph Output[Выход]
        Logits[Logits<br/>vocab_size]
        Code[CadQuery код]
    end
    
    PC --> FPE
    IMG --> VE
    VID --> VE
    TEXT --> TE
    
    FPE --> PCEmbed
    VE --> ImgEmbed
    VE --> VidEmbed
    TE --> Merge
    
    ImgEmbed --> Merge
    VidEmbed --> Merge
    PCEmbed --> Merge
    
    Merge --> RoPE
    RoPE --> Transformer
    Transformer --> LMHead
    LMHead --> Logits
    Logits --> Code
    
    style PC fill:#e1f5ff
    style IMG fill:#fff4e1
    style VID fill:#fff4e1
    style TEXT fill:#f3e5f5
    style FPE fill:#fff4e1
    style VE fill:#e8f5e9
    style Transformer fill:#e8f5e9
    style Code fill:#fce4ec
```

## Сравнение потоков данных

```mermaid
graph LR
    subgraph CADRecodeFlow[CAD-Recode поток]
        A1[STL файл] --> A2[Нормализация]
        A2 --> A3[Point Cloud 256]
        A3 --> A4[Fourier Encoding]
        A4 --> A5[Встраивание через mask]
        A5 --> A6[Qwen2Model]
        A6 --> A7[CadQuery код]
    end
    
    subgraph CadrilleFlow[Cadrille поток]
        B1[Данные] --> B2{Тип?}
        B2 -->|PC| B3[Fourier Encoding]
        B2 -->|IMG| B4[Visual Encoder]
        B2 -->|TEXT| B5[Text Encoder]
        B3 --> B6[Интеграция]
        B4 --> B6
        B5 --> B6
        B6 --> B7[Qwen2VLModel]
        B7 --> B8[CadQuery код]
    end
```

## Детальная архитектура Fourier Point Encoding

```mermaid
graph TD
    subgraph Input[Вход]
        Points[Point Cloud<br/>batch x 256 x 3]
    end
    
    subgraph Fourier[Fourier Encoding]
        Freqs[Частоты<br/>2^0, 2^1, ..., 2^7]
        Mult[Умножение<br/>points * freqs]
        Sin[sin преобразование]
        Cos[cos преобразование]
        Concat[Объединение<br/>points + sin + cos]
    end
    
    subgraph Projection[Проекция]
        Linear[nn.Linear<br/>51 -> hidden_size]
    end
    
    subgraph Output[Выход]
        Embeddings[Point Embeddings<br/>batch x 256 x hidden_size]
    end
    
    Points --> Freqs
    Points --> Mult
    Freqs --> Mult
    Mult --> Sin
    Mult --> Cos
    Points --> Concat
    Sin --> Concat
    Cos --> Concat
    Concat --> Linear
    Linear --> Embeddings
    
    style Points fill:#e1f5ff
    style Concat fill:#fff4e1
    style Linear fill:#e8f5e9
    style Embeddings fill:#fce4ec
```

## Интеграция point embeddings в последовательность

```mermaid
sequenceDiagram
    participant Input as Входные данные
    participant Embed as embed_tokens
    participant PCEnc as FourierPointEncoder
    participant Integrate as Интеграция
    participant Model as Transformer
    
    Note over Input: input_ids = [pad, pad, ..., pad, im_start]
    Note over Input: attention_mask = [-1, -1, ..., -1, 1]
    
    Input->>Embed: Токенизация
    Embed->>Integrate: inputs_embeds (текстовые)
    
    Input->>PCEnc: point_cloud
    PCEnc->>PCEnc: Fourier encoding
    PCEnc->>Integrate: point_embeds
    
    alt CAD-Recode
        Note over Integrate: Векторная операция:<br/>inputs_embeds[mask == -1] = point_embeds
    else Cadrille
        Note over Integrate: Цикл по батчу:<br/>inputs_embeds[i, start_idx:...] = point_embeds[i]
    end
    
    Integrate->>Model: inputs_embeds (объединенные)
    Model->>Model: Обработка через трансформер
    Model-->>Input: hidden_states
```

## Обработка мультимодальных входов в Cadrille

```mermaid
graph TD
    subgraph Inputs[Входные данные]
        PC[Point Cloud]
        IMG[Изображения]
        VID[Видео]
        TEXT[Текст]
    end
    
    subgraph Processing[Обработка]
        PCToken[Pad токены<br/>для PC]
        ImgToken[image_token_id<br/>для изображений]
        VidToken[video_token_id<br/>для видео]
        TextToken[Текстовые токены]
    end
    
    subgraph Encoding[Кодирование]
        PCEnc[FourierPointEncoder]
        ImgEnc[Visual Encoder]
        VidEnc[Visual Encoder]
        TextEnc[embed_tokens]
    end
    
    subgraph Integration[Интеграция]
        PCEmbed[Встраивание PC<br/>start_idxs]
        ImgEmbed[masked_scatter<br/>image_embeds]
        VidEmbed[masked_scatter<br/>video_embeds]
        Merge[Объединение<br/>в inputs_embeds]
    end
    
    PC --> PCToken
    IMG --> ImgToken
    VID --> VidToken
    TEXT --> TextToken
    
    PCToken --> PCEnc
    ImgToken --> ImgEnc
    VidToken --> VidEnc
    TextToken --> TextEnc
    
    PCEnc --> PCEmbed
    ImgEnc --> ImgEmbed
    VidEnc --> VidEmbed
    TextEnc --> Merge
    
    PCEmbed --> Merge
    ImgEmbed --> Merge
    VidEmbed --> Merge
    
    Merge --> Transformer[Qwen2VLModel]
    
    style PC fill:#e1f5ff
    style IMG fill:#fff4e1
    style VID fill:#fff4e1
    style TEXT fill:#f3e5f5
    style Merge fill:#e8f5e9
```
