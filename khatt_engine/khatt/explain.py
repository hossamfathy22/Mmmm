"""Optional: ask Claude to explain, in Arabic, how the uploaded logo was designed.

Needs an Anthropic API key in ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, Optional, Tuple

from PIL import Image

MODEL = os.environ.get("KHATT_MODEL", "claude-opus-5-5")

PROMPT = """أنت مصمم حروف عربية ولاتينية خبير، وتشرح لمصمم يريد أن يتعلّم.
أمامك صورة شعار، وهذه قياسات استخرجها برنامج تحليل بكسلات من الشعار (بالبكسل في الصورة الأصلية):

{analysis}

ثم استُخرجت من السطر رقم {line} هذه القواعد، واستُعملت لتوليد الكلمة «{word}» بنفس الأسلوب:

{style}

معاني الحقول: stem سمك الجذع العمودي، hairline/hair سمك الخط الرفيع، x_height ارتفاع جسم الحروف،
stem_pitch المسافة بين مراكز الجذوع، white الأبيض بين جذعين، counters الفراغات المغلقة، dashes خطوط رفيعة منفصلة.

اكتب بالعربية، بهذه الأقسام:
## الفكرة الأساسية
## الشبكة
## قاعدة القلم
## القطع المشتركة (جدول: القطعة | أين تظهر)
## التصحيحات البصرية
## كيف بُنيت «{word}» من هذه القواعد
## كيف تعيد بناء الشعار في Illustrator (خطوات مرقّمة بالأرقام المقيسة)

القواعد: استعمل الأرقام المقيسة كما هي ولا تخترع أرقامًا جديدة. ما تستنتجه بالنظر دون قياس اكتب بعده (قراءة بصرية).
إن بدا أن البرنامج أخطأ في قياس ما فقل ذلك صراحة. جمل قصيرة وواضحة، بلا مقدمات."""


def _png_b64(data: bytes) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def explain(image: bytes, analysis: Dict[str, Any], style: Dict[str, Any], word: str, line: int) -> Tuple[Optional[str], Optional[str]]:
    """Return (markdown, None) on success or (None, error message in Arabic)."""
    try:
        import anthropic
    except ImportError:
        return None, "مكتبة anthropic غير مثبتة. شغّل: pip install anthropic"
    numbers = {k: v for k, v in analysis.items() if k != "overlay_png"}
    prompt = PROMPT.format(analysis=json.dumps(numbers, ensure_ascii=False, indent=1),
                           style=json.dumps(style, ensure_ascii=False, indent=1), word=word, line=line + 1)
    try:
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(image)}},
                {"type": "text", "text": prompt},
            ]}],
        )
    except anthropic.AuthenticationError:
        return None, "مفتاح Anthropic غير صحيح أو غير موجود. ضع المفتاح في المتغير ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return None, "طلبات كثيرة على Claude الآن. حاول بعد دقيقة."
    except anthropic.APIStatusError as e:
        return None, f"خطأ من خدمة Claude ({e.status_code})."
    except anthropic.APIConnectionError:
        return None, "تعذّر الاتصال بخدمة Claude. تحقّق من الإنترنت."
    except Exception as e:  # missing credentials and similar setup problems
        return None, f"تعذّر تشغيل الشرح: {e}"
    if response.stop_reason == "refusal":
        return None, "رفض Claude هذا الطلب."
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return (text or None), (None if text else "لم يرجع Claude نصًا.")
