# Main Experiment Generation Prompt Catalog

The figure column headers are semantic prompt categories. The `generation_prompt` below is the exact text passed to each method, recovered from the validated image metadata. The clean `evaluation_prompt` is used for CLIP scoring and intentionally excludes `sks` and learned Textual Inversion tokens.

## dog2

### P0: identity

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog` | `a photo of a dog` |
| DreamBooth | `a photo of sks dog` | `a photo of a dog` |
| DreamBooth-LoRA | `a photo of sks dog` | `a photo of a dog` |
| Textual Inversion | `a photo of <sks-dog>` | `a photo of a dog` |
| IP-Adapter Plus | `a photo of a dog` | `a photo of a dog` |

### P1: background_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog on a beach` | `a photo of a dog on a beach` |
| DreamBooth | `a photo of sks dog on a beach` | `a photo of a dog on a beach` |
| DreamBooth-LoRA | `a photo of sks dog on a beach` | `a photo of a dog on a beach` |
| Textual Inversion | `a photo of <sks-dog> on a beach` | `a photo of a dog on a beach` |
| IP-Adapter Plus | `a photo of a dog on a beach` | `a photo of a dog on a beach` |

### P2: context_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |
| DreamBooth | `a photo of sks dog inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |
| DreamBooth-LoRA | `a photo of sks dog inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |
| Textual Inversion | `a photo of <sks-dog> inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |
| IP-Adapter Plus | `a photo of a dog inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |

### P3: attribute_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog with a red accessory` | `a photo of a dog with a red accessory` |
| DreamBooth | `a photo of sks dog with a red accessory` | `a photo of a dog with a red accessory` |
| DreamBooth-LoRA | `a photo of sks dog with a red accessory` | `a photo of a dog with a red accessory` |
| Textual Inversion | `a photo of <sks-dog> with a red accessory` | `a photo of a dog with a red accessory` |
| IP-Adapter Plus | `a photo of a dog with a red accessory` | `a photo of a dog with a red accessory` |

### P4: composition_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog on a wooden table` | `a photo of a dog on a wooden table` |
| DreamBooth | `a photo of sks dog on a wooden table` | `a photo of a dog on a wooden table` |
| DreamBooth-LoRA | `a photo of sks dog on a wooden table` | `a photo of a dog on a wooden table` |
| Textual Inversion | `a photo of <sks-dog> on a wooden table` | `a photo of a dog on a wooden table` |
| IP-Adapter Plus | `a photo of a dog on a wooden table` | `a photo of a dog on a wooden table` |

### P5: style_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a watercolor painting of a dog` | `a watercolor painting of a dog` |
| DreamBooth | `a watercolor painting of sks dog` | `a watercolor painting of a dog` |
| DreamBooth-LoRA | `a watercolor painting of sks dog` | `a watercolor painting of a dog` |
| Textual Inversion | `a watercolor painting of <sks-dog>` | `a watercolor painting of a dog` |
| IP-Adapter Plus | `a watercolor painting of a dog` | `a watercolor painting of a dog` |

## cat

### P0: identity

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a cat` | `a photo of a cat` |
| DreamBooth | `a photo of sks cat` | `a photo of a cat` |
| DreamBooth-LoRA | `a photo of sks cat` | `a photo of a cat` |
| Textual Inversion | `a photo of <sks-cat>` | `a photo of a cat` |
| IP-Adapter Plus | `a photo of a cat` | `a photo of a cat` |

### P1: background_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a cat on a beach` | `a photo of a cat on a beach` |
| DreamBooth | `a photo of sks cat on a beach` | `a photo of a cat on a beach` |
| DreamBooth-LoRA | `a photo of sks cat on a beach` | `a photo of a cat on a beach` |
| Textual Inversion | `a photo of <sks-cat> on a beach` | `a photo of a cat on a beach` |
| IP-Adapter Plus | `a photo of a cat on a beach` | `a photo of a cat on a beach` |

### P2: context_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a cat inside a modern kitchen` | `a photo of a cat inside a modern kitchen` |
| DreamBooth | `a photo of sks cat inside a modern kitchen` | `a photo of a cat inside a modern kitchen` |
| DreamBooth-LoRA | `a photo of sks cat inside a modern kitchen` | `a photo of a cat inside a modern kitchen` |
| Textual Inversion | `a photo of <sks-cat> inside a modern kitchen` | `a photo of a cat inside a modern kitchen` |
| IP-Adapter Plus | `a photo of a cat inside a modern kitchen` | `a photo of a cat inside a modern kitchen` |

### P3: attribute_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a cat with a red accessory` | `a photo of a cat with a red accessory` |
| DreamBooth | `a photo of sks cat with a red accessory` | `a photo of a cat with a red accessory` |
| DreamBooth-LoRA | `a photo of sks cat with a red accessory` | `a photo of a cat with a red accessory` |
| Textual Inversion | `a photo of <sks-cat> with a red accessory` | `a photo of a cat with a red accessory` |
| IP-Adapter Plus | `a photo of a cat with a red accessory` | `a photo of a cat with a red accessory` |

### P4: composition_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a cat on a wooden table` | `a photo of a cat on a wooden table` |
| DreamBooth | `a photo of sks cat on a wooden table` | `a photo of a cat on a wooden table` |
| DreamBooth-LoRA | `a photo of sks cat on a wooden table` | `a photo of a cat on a wooden table` |
| Textual Inversion | `a photo of <sks-cat> on a wooden table` | `a photo of a cat on a wooden table` |
| IP-Adapter Plus | `a photo of a cat on a wooden table` | `a photo of a cat on a wooden table` |

### P5: style_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a watercolor painting of a cat` | `a watercolor painting of a cat` |
| DreamBooth | `a watercolor painting of sks cat` | `a watercolor painting of a cat` |
| DreamBooth-LoRA | `a watercolor painting of sks cat` | `a watercolor painting of a cat` |
| Textual Inversion | `a watercolor painting of <sks-cat>` | `a watercolor painting of a cat` |
| IP-Adapter Plus | `a watercolor painting of a cat` | `a watercolor painting of a cat` |

## pet_dog1

### P0: identity

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog` | `a photo of a dog` |
| DreamBooth | `a photo of sks dog` | `a photo of a dog` |
| DreamBooth-LoRA | `a photo of sks dog` | `a photo of a dog` |
| Textual Inversion | `a photo of <sks-dog-pet1>` | `a photo of a dog` |
| IP-Adapter Plus | `a photo of a dog` | `a photo of a dog` |

### P1: background_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog on a beach` | `a photo of a dog on a beach` |
| DreamBooth | `a photo of sks dog on a beach` | `a photo of a dog on a beach` |
| DreamBooth-LoRA | `a photo of sks dog on a beach` | `a photo of a dog on a beach` |
| Textual Inversion | `a photo of <sks-dog-pet1> on a beach` | `a photo of a dog on a beach` |
| IP-Adapter Plus | `a photo of a dog on a beach` | `a photo of a dog on a beach` |

### P2: context_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |
| DreamBooth | `a photo of sks dog inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |
| DreamBooth-LoRA | `a photo of sks dog inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |
| Textual Inversion | `a photo of <sks-dog-pet1> inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |
| IP-Adapter Plus | `a photo of a dog inside a modern kitchen` | `a photo of a dog inside a modern kitchen` |

### P3: attribute_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog with a red accessory` | `a photo of a dog with a red accessory` |
| DreamBooth | `a photo of sks dog with a red accessory` | `a photo of a dog with a red accessory` |
| DreamBooth-LoRA | `a photo of sks dog with a red accessory` | `a photo of a dog with a red accessory` |
| Textual Inversion | `a photo of <sks-dog-pet1> with a red accessory` | `a photo of a dog with a red accessory` |
| IP-Adapter Plus | `a photo of a dog with a red accessory` | `a photo of a dog with a red accessory` |

### P4: composition_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a dog on a wooden table` | `a photo of a dog on a wooden table` |
| DreamBooth | `a photo of sks dog on a wooden table` | `a photo of a dog on a wooden table` |
| DreamBooth-LoRA | `a photo of sks dog on a wooden table` | `a photo of a dog on a wooden table` |
| Textual Inversion | `a photo of <sks-dog-pet1> on a wooden table` | `a photo of a dog on a wooden table` |
| IP-Adapter Plus | `a photo of a dog on a wooden table` | `a photo of a dog on a wooden table` |

### P5: style_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a watercolor painting of a dog` | `a watercolor painting of a dog` |
| DreamBooth | `a watercolor painting of sks dog` | `a watercolor painting of a dog` |
| DreamBooth-LoRA | `a watercolor painting of sks dog` | `a watercolor painting of a dog` |
| Textual Inversion | `a watercolor painting of <sks-dog-pet1>` | `a watercolor painting of a dog` |
| IP-Adapter Plus | `a watercolor painting of a dog` | `a watercolor painting of a dog` |

## bear_plushie

### P0: identity

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a stuffed animal` | `a photo of a stuffed animal` |
| DreamBooth | `a photo of sks stuffed animal` | `a photo of a stuffed animal` |
| DreamBooth-LoRA | `a photo of sks stuffed animal` | `a photo of a stuffed animal` |
| Textual Inversion | `a photo of <sks-bear-plushie>` | `a photo of a stuffed animal` |
| IP-Adapter Plus | `a photo of a stuffed animal` | `a photo of a stuffed animal` |

### P1: background_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a stuffed animal on a beach` | `a photo of a stuffed animal on a beach` |
| DreamBooth | `a photo of sks stuffed animal on a beach` | `a photo of a stuffed animal on a beach` |
| DreamBooth-LoRA | `a photo of sks stuffed animal on a beach` | `a photo of a stuffed animal on a beach` |
| Textual Inversion | `a photo of <sks-bear-plushie> on a beach` | `a photo of a stuffed animal on a beach` |
| IP-Adapter Plus | `a photo of a stuffed animal on a beach` | `a photo of a stuffed animal on a beach` |

### P2: context_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a stuffed animal inside a modern kitchen` | `a photo of a stuffed animal inside a modern kitchen` |
| DreamBooth | `a photo of sks stuffed animal inside a modern kitchen` | `a photo of a stuffed animal inside a modern kitchen` |
| DreamBooth-LoRA | `a photo of sks stuffed animal inside a modern kitchen` | `a photo of a stuffed animal inside a modern kitchen` |
| Textual Inversion | `a photo of <sks-bear-plushie> inside a modern kitchen` | `a photo of a stuffed animal inside a modern kitchen` |
| IP-Adapter Plus | `a photo of a stuffed animal inside a modern kitchen` | `a photo of a stuffed animal inside a modern kitchen` |

### P3: attribute_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a stuffed animal with a red accessory` | `a photo of a stuffed animal with a red accessory` |
| DreamBooth | `a photo of sks stuffed animal with a red accessory` | `a photo of a stuffed animal with a red accessory` |
| DreamBooth-LoRA | `a photo of sks stuffed animal with a red accessory` | `a photo of a stuffed animal with a red accessory` |
| Textual Inversion | `a photo of <sks-bear-plushie> with a red accessory` | `a photo of a stuffed animal with a red accessory` |
| IP-Adapter Plus | `a photo of a stuffed animal with a red accessory` | `a photo of a stuffed animal with a red accessory` |

### P4: composition_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a stuffed animal on a wooden table` | `a photo of a stuffed animal on a wooden table` |
| DreamBooth | `a photo of sks stuffed animal on a wooden table` | `a photo of a stuffed animal on a wooden table` |
| DreamBooth-LoRA | `a photo of sks stuffed animal on a wooden table` | `a photo of a stuffed animal on a wooden table` |
| Textual Inversion | `a photo of <sks-bear-plushie> on a wooden table` | `a photo of a stuffed animal on a wooden table` |
| IP-Adapter Plus | `a photo of a stuffed animal on a wooden table` | `a photo of a stuffed animal on a wooden table` |

### P5: style_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a watercolor painting of a stuffed animal` | `a watercolor painting of a stuffed animal` |
| DreamBooth | `a watercolor painting of sks stuffed animal` | `a watercolor painting of a stuffed animal` |
| DreamBooth-LoRA | `a watercolor painting of sks stuffed animal` | `a watercolor painting of a stuffed animal` |
| Textual Inversion | `a watercolor painting of <sks-bear-plushie>` | `a watercolor painting of a stuffed animal` |
| IP-Adapter Plus | `a watercolor painting of a stuffed animal` | `a watercolor painting of a stuffed animal` |

## robot_toy

### P0: identity

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a toy` | `a photo of a toy` |
| DreamBooth | `a photo of sks toy` | `a photo of a toy` |
| DreamBooth-LoRA | `a photo of sks toy` | `a photo of a toy` |
| Textual Inversion | `a photo of <sks-robot-toy>` | `a photo of a toy` |
| IP-Adapter Plus | `a photo of a toy` | `a photo of a toy` |

### P1: background_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a toy on a beach` | `a photo of a toy on a beach` |
| DreamBooth | `a photo of sks toy on a beach` | `a photo of a toy on a beach` |
| DreamBooth-LoRA | `a photo of sks toy on a beach` | `a photo of a toy on a beach` |
| Textual Inversion | `a photo of <sks-robot-toy> on a beach` | `a photo of a toy on a beach` |
| IP-Adapter Plus | `a photo of a toy on a beach` | `a photo of a toy on a beach` |

### P2: context_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a toy inside a modern kitchen` | `a photo of a toy inside a modern kitchen` |
| DreamBooth | `a photo of sks toy inside a modern kitchen` | `a photo of a toy inside a modern kitchen` |
| DreamBooth-LoRA | `a photo of sks toy inside a modern kitchen` | `a photo of a toy inside a modern kitchen` |
| Textual Inversion | `a photo of <sks-robot-toy> inside a modern kitchen` | `a photo of a toy inside a modern kitchen` |
| IP-Adapter Plus | `a photo of a toy inside a modern kitchen` | `a photo of a toy inside a modern kitchen` |

### P3: attribute_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a toy with a red accessory` | `a photo of a toy with a red accessory` |
| DreamBooth | `a photo of sks toy with a red accessory` | `a photo of a toy with a red accessory` |
| DreamBooth-LoRA | `a photo of sks toy with a red accessory` | `a photo of a toy with a red accessory` |
| Textual Inversion | `a photo of <sks-robot-toy> with a red accessory` | `a photo of a toy with a red accessory` |
| IP-Adapter Plus | `a photo of a toy with a red accessory` | `a photo of a toy with a red accessory` |

### P4: composition_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a toy on a wooden table` | `a photo of a toy on a wooden table` |
| DreamBooth | `a photo of sks toy on a wooden table` | `a photo of a toy on a wooden table` |
| DreamBooth-LoRA | `a photo of sks toy on a wooden table` | `a photo of a toy on a wooden table` |
| Textual Inversion | `a photo of <sks-robot-toy> on a wooden table` | `a photo of a toy on a wooden table` |
| IP-Adapter Plus | `a photo of a toy on a wooden table` | `a photo of a toy on a wooden table` |

### P5: style_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a watercolor painting of a toy` | `a watercolor painting of a toy` |
| DreamBooth | `a watercolor painting of sks toy` | `a watercolor painting of a toy` |
| DreamBooth-LoRA | `a watercolor painting of sks toy` | `a watercolor painting of a toy` |
| Textual Inversion | `a watercolor painting of <sks-robot-toy>` | `a watercolor painting of a toy` |
| IP-Adapter Plus | `a watercolor painting of a toy` | `a watercolor painting of a toy` |

## teapot

### P0: identity

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a teapot` | `a photo of a teapot` |
| DreamBooth | `a photo of sks teapot` | `a photo of a teapot` |
| DreamBooth-LoRA | `a photo of sks teapot` | `a photo of a teapot` |
| Textual Inversion | `a photo of <sks-teapot>` | `a photo of a teapot` |
| IP-Adapter Plus | `a photo of a teapot` | `a photo of a teapot` |

### P1: background_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a teapot on a beach` | `a photo of a teapot on a beach` |
| DreamBooth | `a photo of sks teapot on a beach` | `a photo of a teapot on a beach` |
| DreamBooth-LoRA | `a photo of sks teapot on a beach` | `a photo of a teapot on a beach` |
| Textual Inversion | `a photo of <sks-teapot> on a beach` | `a photo of a teapot on a beach` |
| IP-Adapter Plus | `a photo of a teapot on a beach` | `a photo of a teapot on a beach` |

### P2: context_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a teapot inside a modern kitchen` | `a photo of a teapot inside a modern kitchen` |
| DreamBooth | `a photo of sks teapot inside a modern kitchen` | `a photo of a teapot inside a modern kitchen` |
| DreamBooth-LoRA | `a photo of sks teapot inside a modern kitchen` | `a photo of a teapot inside a modern kitchen` |
| Textual Inversion | `a photo of <sks-teapot> inside a modern kitchen` | `a photo of a teapot inside a modern kitchen` |
| IP-Adapter Plus | `a photo of a teapot inside a modern kitchen` | `a photo of a teapot inside a modern kitchen` |

### P3: attribute_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a teapot with a red accessory` | `a photo of a teapot with a red accessory` |
| DreamBooth | `a photo of sks teapot with a red accessory` | `a photo of a teapot with a red accessory` |
| DreamBooth-LoRA | `a photo of sks teapot with a red accessory` | `a photo of a teapot with a red accessory` |
| Textual Inversion | `a photo of <sks-teapot> with a red accessory` | `a photo of a teapot with a red accessory` |
| IP-Adapter Plus | `a photo of a teapot with a red accessory` | `a photo of a teapot with a red accessory` |

### P4: composition_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a teapot on a wooden table` | `a photo of a teapot on a wooden table` |
| DreamBooth | `a photo of sks teapot on a wooden table` | `a photo of a teapot on a wooden table` |
| DreamBooth-LoRA | `a photo of sks teapot on a wooden table` | `a photo of a teapot on a wooden table` |
| Textual Inversion | `a photo of <sks-teapot> on a wooden table` | `a photo of a teapot on a wooden table` |
| IP-Adapter Plus | `a photo of a teapot on a wooden table` | `a photo of a teapot on a wooden table` |

### P5: style_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a watercolor painting of a teapot` | `a watercolor painting of a teapot` |
| DreamBooth | `a watercolor painting of sks teapot` | `a watercolor painting of a teapot` |
| DreamBooth-LoRA | `a watercolor painting of sks teapot` | `a watercolor painting of a teapot` |
| Textual Inversion | `a watercolor painting of <sks-teapot>` | `a watercolor painting of a teapot` |
| IP-Adapter Plus | `a watercolor painting of a teapot` | `a watercolor painting of a teapot` |

## colorful_sneaker

### P0: identity

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a sneaker` | `a photo of a sneaker` |
| DreamBooth | `a photo of sks sneaker` | `a photo of a sneaker` |
| DreamBooth-LoRA | `a photo of sks sneaker` | `a photo of a sneaker` |
| Textual Inversion | `a photo of <sks-colorful-sneaker>` | `a photo of a sneaker` |
| IP-Adapter Plus | `a photo of a sneaker` | `a photo of a sneaker` |

### P1: background_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a sneaker on a beach` | `a photo of a sneaker on a beach` |
| DreamBooth | `a photo of sks sneaker on a beach` | `a photo of a sneaker on a beach` |
| DreamBooth-LoRA | `a photo of sks sneaker on a beach` | `a photo of a sneaker on a beach` |
| Textual Inversion | `a photo of <sks-colorful-sneaker> on a beach` | `a photo of a sneaker on a beach` |
| IP-Adapter Plus | `a photo of a sneaker on a beach` | `a photo of a sneaker on a beach` |

### P2: context_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a sneaker inside a modern kitchen` | `a photo of a sneaker inside a modern kitchen` |
| DreamBooth | `a photo of sks sneaker inside a modern kitchen` | `a photo of a sneaker inside a modern kitchen` |
| DreamBooth-LoRA | `a photo of sks sneaker inside a modern kitchen` | `a photo of a sneaker inside a modern kitchen` |
| Textual Inversion | `a photo of <sks-colorful-sneaker> inside a modern kitchen` | `a photo of a sneaker inside a modern kitchen` |
| IP-Adapter Plus | `a photo of a sneaker inside a modern kitchen` | `a photo of a sneaker inside a modern kitchen` |

### P3: attribute_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a sneaker with a red accessory` | `a photo of a sneaker with a red accessory` |
| DreamBooth | `a photo of sks sneaker with a red accessory` | `a photo of a sneaker with a red accessory` |
| DreamBooth-LoRA | `a photo of sks sneaker with a red accessory` | `a photo of a sneaker with a red accessory` |
| Textual Inversion | `a photo of <sks-colorful-sneaker> with a red accessory` | `a photo of a sneaker with a red accessory` |
| IP-Adapter Plus | `a photo of a sneaker with a red accessory` | `a photo of a sneaker with a red accessory` |

### P4: composition_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a sneaker on a wooden table` | `a photo of a sneaker on a wooden table` |
| DreamBooth | `a photo of sks sneaker on a wooden table` | `a photo of a sneaker on a wooden table` |
| DreamBooth-LoRA | `a photo of sks sneaker on a wooden table` | `a photo of a sneaker on a wooden table` |
| Textual Inversion | `a photo of <sks-colorful-sneaker> on a wooden table` | `a photo of a sneaker on a wooden table` |
| IP-Adapter Plus | `a photo of a sneaker on a wooden table` | `a photo of a sneaker on a wooden table` |

### P5: style_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a watercolor painting of a sneaker` | `a watercolor painting of a sneaker` |
| DreamBooth | `a watercolor painting of sks sneaker` | `a watercolor painting of a sneaker` |
| DreamBooth-LoRA | `a watercolor painting of sks sneaker` | `a watercolor painting of a sneaker` |
| Textual Inversion | `a watercolor painting of <sks-colorful-sneaker>` | `a watercolor painting of a sneaker` |
| IP-Adapter Plus | `a watercolor painting of a sneaker` | `a watercolor painting of a sneaker` |

## luggage_backpack1

### P0: identity

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a backpack` | `a photo of a backpack` |
| DreamBooth | `a photo of sks backpack` | `a photo of a backpack` |
| DreamBooth-LoRA | `a photo of sks backpack` | `a photo of a backpack` |
| Textual Inversion | `a photo of <sks-backpack>` | `a photo of a backpack` |
| IP-Adapter Plus | `a photo of a backpack` | `a photo of a backpack` |

### P1: background_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a backpack on a beach` | `a photo of a backpack on a beach` |
| DreamBooth | `a photo of sks backpack on a beach` | `a photo of a backpack on a beach` |
| DreamBooth-LoRA | `a photo of sks backpack on a beach` | `a photo of a backpack on a beach` |
| Textual Inversion | `a photo of <sks-backpack> on a beach` | `a photo of a backpack on a beach` |
| IP-Adapter Plus | `a photo of a backpack on a beach` | `a photo of a backpack on a beach` |

### P2: context_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a backpack inside a modern kitchen` | `a photo of a backpack inside a modern kitchen` |
| DreamBooth | `a photo of sks backpack inside a modern kitchen` | `a photo of a backpack inside a modern kitchen` |
| DreamBooth-LoRA | `a photo of sks backpack inside a modern kitchen` | `a photo of a backpack inside a modern kitchen` |
| Textual Inversion | `a photo of <sks-backpack> inside a modern kitchen` | `a photo of a backpack inside a modern kitchen` |
| IP-Adapter Plus | `a photo of a backpack inside a modern kitchen` | `a photo of a backpack inside a modern kitchen` |

### P3: attribute_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a backpack with a red accessory` | `a photo of a backpack with a red accessory` |
| DreamBooth | `a photo of sks backpack with a red accessory` | `a photo of a backpack with a red accessory` |
| DreamBooth-LoRA | `a photo of sks backpack with a red accessory` | `a photo of a backpack with a red accessory` |
| Textual Inversion | `a photo of <sks-backpack> with a red accessory` | `a photo of a backpack with a red accessory` |
| IP-Adapter Plus | `a photo of a backpack with a red accessory` | `a photo of a backpack with a red accessory` |

### P4: composition_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a photo of a backpack on a wooden table` | `a photo of a backpack on a wooden table` |
| DreamBooth | `a photo of sks backpack on a wooden table` | `a photo of a backpack on a wooden table` |
| DreamBooth-LoRA | `a photo of sks backpack on a wooden table` | `a photo of a backpack on a wooden table` |
| Textual Inversion | `a photo of <sks-backpack> on a wooden table` | `a photo of a backpack on a wooden table` |
| IP-Adapter Plus | `a photo of a backpack on a wooden table` | `a photo of a backpack on a wooden table` |

### P5: style_change

| Method | Exact generation prompt | Clean evaluation prompt |
|---|---|---|
| Base SD1.5
(no personalization) | `a watercolor painting of a backpack` | `a watercolor painting of a backpack` |
| DreamBooth | `a watercolor painting of sks backpack` | `a watercolor painting of a backpack` |
| DreamBooth-LoRA | `a watercolor painting of sks backpack` | `a watercolor painting of a backpack` |
| Textual Inversion | `a watercolor painting of <sks-backpack>` | `a watercolor painting of a backpack` |
| IP-Adapter Plus | `a watercolor painting of a backpack` | `a watercolor painting of a backpack` |

