"""matchum (맞춤) — 규칙 기반 오프라인 한국어 맞춤법 검사 라이브러리.

    >>> from matchum import check
    >>> check("몇일 뒤에 연락할께요").checked
    '며칠 뒤에 연락할게요'

MIT License. 있는 그대로(AS IS) 제공되며 어떠한 보증도 하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


# ============================================================================
# 한글 음절 유틸리티
# ============================================================================

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

_BASE = 0xAC00
_LAST = 0xD7A3


def is_syllable(ch: str) -> bool:
    """완성형 한글 음절 한 글자인지."""
    return len(ch) == 1 and _BASE <= ord(ch) <= _LAST


def decompose(ch: str) -> Optional[Tuple[str, str, str]]:
    """음절을 (초성, 중성, 종성)으로 분해. 한글 음절이 아니면 None."""
    if not is_syllable(ch):
        return None
    code = ord(ch) - _BASE
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def compose(cho: str, jung: str, jong: str = "") -> str:
    """(초성, 중성, 종성)을 음절로 조합."""
    return chr(_BASE + CHO.index(cho) * 588 + JUNG.index(jung) * 28 + JONG.index(jong))


def jongseong(ch: str) -> str:
    """종성(받침). 받침이 없거나 한글이 아니면 빈 문자열."""
    parts = decompose(ch)
    return parts[2] if parts else ""


def has_jongseong(ch: str) -> bool:
    return bool(jongseong(ch))


# ============================================================================
# 결과 자료형
# ============================================================================

class Level(str, Enum):
    ERROR = "error"      # 문맥과 무관하게 틀린 표기 — 기본적으로 자동 교정
    WARNING = "warning"  # 문맥에 따라 맞을 수도 있는 표기 — 기본적으로 보고만 함


class Category(str, Enum):
    SPELLING = "spelling"  # 맞춤법
    SPACING = "spacing"    # 띄어쓰기
    STYLE = "style"        # 표준어·문체 권고


@dataclass(frozen=True)
class Correction:
    start: int          # 원문 기준 시작 인덱스
    end: int            # 원문 기준 끝 인덱스 (exclusive)
    original: str
    suggestion: str
    rule_id: str
    message: str
    level: Level = Level.ERROR
    category: Category = Category.SPELLING


@dataclass
class Result:
    original: str
    checked: str
    corrections: List[Correction] = field(default_factory=list)
    time: float = 0.0

    @property
    def errors(self) -> int:
        return len(self.corrections)

    @property
    def ok(self) -> bool:
        return not self.corrections

    def as_dict(self) -> Dict:
        return {
            "original": self.original,
            "checked": self.checked,
            "errors": self.errors,
            "time": self.time,
            "corrections": [
                {
                    "start": c.start,
                    "end": c.end,
                    "original": c.original,
                    "suggestion": c.suggestion,
                    "rule": c.rule_id,
                    "message": c.message,
                    "level": c.level.value,
                    "category": c.category.value,
                }
                for c in self.corrections
            ],
        }


# ============================================================================
# 규칙 베이스
# ============================================================================
#
# 규칙 베이스 클래스.
#
# 규칙 하나 = 오류 패턴 하나. 새 규칙은 아래 세 가지 중 하나로 만든다.
#   - WordRule  : 항상 틀린 고정 표기 → 고정 교정어 (사전형, 대부분의 규칙)
#   - RegexRule : 정규식 + 교정 함수 (어미·띄어쓰기 등 패턴형)
#   - Rule      : 직접 find() 구현 (형태소 분석 등이 필요한 고급 규칙)

# 한글 음절 앞에 붙어 있지 않을 것 (단어 시작 경계)
WORD_START = r"(?<![가-힣])"


class Rule(ABC):
    id: str = ""
    message: str = ""
    level: Level = Level.ERROR
    category: Category = Category.SPELLING

    @abstractmethod
    def find(self, text: str) -> Iterator[Correction]:
        ...

    def _make(self, start: int, end: int, text: str, suggestion: str,
              message: Optional[str] = None) -> Correction:
        return Correction(
            start=start, end=end, original=text[start:end], suggestion=suggestion,
            rule_id=self.id, message=message or self.message,
            level=self.level, category=self.category,
        )


def _covered(text: str, start: int, end: int, word: str) -> bool:
    """text 안에서 word가 [start, end) 구간을 통째로 덮고, 단어 첫머리에서 시작하는지.

    '날것'은 예외지만 '끝날것'의 '날것'은 예외가 아니다.
    """
    for st in range(max(0, end - len(word)), start + 1):
        if text.startswith(word, st) and (st == 0 or not _is_hangul(text[st - 1])):
            return True
    return False


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def _iga(word: str) -> str:
    """끝 글자 받침에 맞는 주격 조사 (이/가)."""
    last = word.rstrip()[-1:] if word.strip() else ""
    return "이" if last and has_jongseong(last) else "가"


class RegexRule(Rule):
    """정규식 매치마다 fix(match)로 교정어를 만든다. fix가 None을 반환하면 건너뜀.

    exceptions: 매치 구간을 덮는 정상 단어 목록 (예: '것' 띄어쓰기 규칙의 '날것', '들것').
    """

    def __init__(self, id: str, pattern: str, fix: Callable[[re.Match], Optional[str]],
                 message: str, level: Level = Level.ERROR,
                 category: Category = Category.SPELLING,
                 exceptions: Iterable[str] = ()):
        self.id = id
        self.pattern = re.compile(pattern)
        self.fix = fix
        self.message = message
        self.level = level
        self.category = category
        self.exceptions = tuple(exceptions)

    def find(self, text: str) -> Iterator[Correction]:
        for m in self.pattern.finditer(text):
            if any(_covered(text, m.start(), m.end(), w) for w in self.exceptions):
                continue
            suggestion = self.fix(m)
            if suggestion is not None and suggestion != m.group(0):
                yield self._make(m.start(), m.end(), text, suggestion)


class WordRule(RegexRule):
    """고정 오표기 → 고정 교정어. word_start=True면 단어 첫머리에서만 잡는다(합성어 오탐 방지)."""

    def __init__(self, id: str, wrong: str, right: str, message: str = "",
                 word_start: bool = False, level: Level = Level.ERROR,
                 category: Category = Category.SPELLING,
                 exceptions: Iterable[str] = ()):
        prefix = WORD_START if word_start else ""
        self.wrong, self.right = wrong, right
        super().__init__(
            id=id,
            pattern=prefix + re.escape(wrong),
            fix=lambda m, r=right: r,
            message=message or f"'{right}'{_iga(right)} 맞는 표기입니다.",
            level=level,
            category=category,
            exceptions=exceptions,
        )


def word_rules(prefix: str, table: Iterable[tuple], **kwargs) -> list:
    """(오표기, 교정어[, 옵션dict]) 표를 WordRule 목록으로 변환."""
    rules = []
    for i, row in enumerate(table, 1):
        wrong, right, *rest = row
        opts = {**kwargs, **(rest[0] if rest else {})}
        rules.append(WordRule(f"{prefix}.{i:03d}", wrong, right, **opts))
    return rules


# ============================================================================
# 사전형 규칙
# ============================================================================
#
# 사전형 규칙: 고정 오표기 → 고정 교정어.
#
# 행 하나가 규칙 하나다. 형식: (오표기, 교정어[, 옵션])
#   옵션 키: word_start(단어 첫머리에서만), exceptions(매치를 덮는 정상 단어), message
#
# 기준은 현행 한글 맞춤법·표준어 규정·외래어 표기법. 교정 쌍과 메시지는 직접 작성한다
# (외부 사전·자료의 해설문이나 예문을 복사하지 않는다).
#
# 새 행을 넣을 때 확인할 것
#   1. 오표기가 다른 정상 단어의 일부로 나오지 않는가 → 나오면 word_start / exceptions
#   2. 문맥에 따라 맞을 수도 있는가 → 그렇다면 _CONTEXT 표(WARNING)에 넣는다
#   3. tests/ 에 교정 케이스와 오탐 방지 케이스를 추가한다

WS = {"word_start": True}

# ---------------------------------------------------------------------------
# 되/돼, 활용형
# ---------------------------------------------------------------------------
_DOE = [
    ("됬", "됐"),
    ("됀", "된"),
    ("됄", "될"),
    ("되요", "돼요"),
    ("되서", "돼서"),
    ("되야", "돼야"),
    ("돼었", "되었"),
    ("돼고", "되고"),
    ("돼는", "되는"),
    ("돼면", "되면"),
    ("돼니", "되니"),
    ("않되", "안 되"),
    ("뵈요", "봬요"),
    ("뵜", "뵀"),
    ("사겼", "사귀었"),
    ("바꼈", "바뀌었"),
    ("헤매이", "헤매"),
    ("되뇌이", "되뇌"),
    ("설레이", "설레"),
    ("끼여들", "끼어들"),
    ("알맞는", "알맞은"),
    ("걸맞는", "걸맞은"),
    ("삼가해", "삼가"),
    ("삼가하", "삼가"),
    ("꺽어", "꺾어"),
    ("꺽었", "꺾었"),
    ("꺽인", "꺾인"),
    ("꺽고", "꺾고"),
    ("들어났", "드러났"),
    ("무릎쓰", "무릅쓰"),
    ("쑥쓰럽", "쑥스럽"),
    ("애띤", "앳된"),
    ("널부러", "널브러"),
    ("희안하", "희한하"),
    ("희안한", "희한한"),
    ("어의없", "어이없"),
    ("어의가 없", "어이가 없"),
    ("어줍잖", "어쭙잖"),
    ("닥달", "닦달"),
    ("핼쓱", "핼쑥"),
    ("아니예요", "아니에요"),
    ("아니였", "아니었"),
    ("십시요", "십시오"),
    ("슴니다", "습니다"),
    ("읍니다", "습니다"),
    ("거에요", "거예요"),
    ("뭐에요", "뭐예요"),
]

# ---------------------------------------------------------------------------
# 부사 -이/-히, 기타 부사
# ---------------------------------------------------------------------------
_ADVERB = [
    ("일일히", "일일이"),
    ("깨끗히", "깨끗이"),
    ("곰곰히", "곰곰이"),
    ("틈틈히", "틈틈이"),
    ("번번히", "번번이"),
    ("꼼꼼이", "꼼꼼히"),
    ("솔직이", "솔직히"),
    ("가만이", "가만히"),
    ("일찌기", "일찍이"),
    ("더우기", "더욱이"),
    ("생각컨데", "생각건대"),
    ("요컨데", "요컨대"),
    ("바라건데", "바라건대"),
    ("비로서", "비로소"),
    ("금새", "금세", {"word_start": True, "exceptions": ["금새우"]}),
    ("괜시리", "괜스레"),
    ("아뭏든", "아무튼"),
    ("하옇든", "하여튼"),
    ("어쨋든", "어쨌든"),
    ("어떻든간에", "어떻든 간에"),
    ("오랫만", "오랜만"),
    ("오랜동안", "오랫동안"),
    ("엇그제", "엊그제"),
    ("웬지", "왠지"),
    ("왠일", "웬일"),
    ("왠만", "웬만"),
    ("어떻해", "어떡해"),
    ("어떻하", "어떡하"),
    ("어떻케", "어떻게"),
    ("어떡게", "어떻게"),
    ("몇일", "며칠"),
    ("몇 일", "며칠"),
    ("어물쩡", "어물쩍"),
]

# ---------------------------------------------------------------------------
# 명사·어휘
# ---------------------------------------------------------------------------
_NOUN = [
    ("역활", "역할"),
    ("설겆이", "설거지"),
    ("내노라", "내로라"),
    ("안밖", "안팎", WS),
    ("우뢰", "우레", WS),
    ("짜집기", "짜깁기"),
    ("짜투리", "자투리"),
    ("뒤치닥거리", "뒤치다꺼리"),
    ("귀뜸", "귀띔"),
    ("눈쌀", "눈살"),
    ("오지랍", "오지랖"),
    ("쑥맥", "숙맥"),
    ("쉽상", "십상"),
    ("폭팔", "폭발"),
    ("뇌졸증", "뇌졸중"),
    ("구렛나루", "구레나룻"),
    ("귓볼", "귓불"),
    ("굼뱅이", "굼벵이"),
    ("개거품", "게거품"),
    ("제작년", "재작년"),
    ("육계장", "육개장"),
    ("김치찌게", "김치찌개"),
    ("된장찌게", "된장찌개"),
    ("부대찌게", "부대찌개"),
    ("순두부찌게", "순두부찌개"),
    ("떡볶기", "떡볶이"),
    ("깍뚜기", "깍두기"),
    ("곱배기", "곱빼기"),
    ("쭈꾸미", "주꾸미"),
    ("어리버리", "어리바리"),
    # 사이시옷
    ("윗층", "위층"),
    ("윗쪽", "위쪽"),
    ("뒷쪽", "뒤쪽"),
    ("뒷풀이", "뒤풀이"),
    ("뒷처리", "뒤처리"),
    ("뒷편", "뒤편"),
    ("윗어른", "웃어른"),
    ("햇님", "해님"),
    ("갯수", "개수"),
    ("촛점", "초점"),
    ("홧병", "화병"),
    ("싯가", "시가"),
    ("머릿말", "머리말"),
    ("인삿말", "인사말"),
    # 암수
    ("숫놈", "수놈"),
    ("숫닭", "수탉"),
    ("숫캐", "수캐"),
    ("숫소", "수소"),
    # 한자 성어
    ("희노애락", "희로애락"),
    ("홀홀단신", "혈혈단신"),
    ("산수갑산", "삼수갑산"),
    ("절대절명", "절체절명"),
    ("동거동락", "동고동락"),
    ("야밤도주", "야반도주"),
    ("토사광란", "토사곽란"),
    ("유도심문", "유도신문"),
    ("염치불구", "염치 불고"),
    ("일사분란", "일사불란"),
    ("포복졸도", "포복절도"),
    ("성대묘사", "성대모사"),
    ("환골탈퇴", "환골탈태"),
    ("고정관렴", "고정관념"),
    # -율/-률 (모음·ㄴ 뒤 '율', 그 밖 '률')
    ("확율", "확률"),
    ("비률", "비율"),
    ("합격율", "합격률"),
    ("성공율", "성공률"),
    ("출석율", "출석률"),
    ("경쟁율", "경쟁률"),
    ("불량율", "불량률"),
    ("시청율", "시청률"),
    ("점유률", "점유율"),
    ("이자률", "이자율"),
    ("할인률", "할인율"),
    ("백분률", "백분율"),
    ("에매", "애매", WS),
]

# ---------------------------------------------------------------------------
# 외래어 표기 (현장 표기가 굳은 경우가 많아 WARNING)
# ---------------------------------------------------------------------------
_LOANWORD = [
    ("메세지", "메시지"),
    ("악세사리", "액세서리"),
    ("악세서리", "액세서리"),
    ("컨텐츠", "콘텐츠"),
    ("쥬스", "주스"),
    ("쵸콜릿", "초콜릿"),
    ("초콜렛", "초콜릿"),
    ("케익", "케이크"),
    ("까페", "카페"),
    ("화이팅", "파이팅"),
    ("리더쉽", "리더십"),
    ("멤버쉽", "멤버십"),
    ("파트너쉽", "파트너십"),
    ("인턴쉽", "인턴십"),
    ("어플리케이션", "애플리케이션"),
    ("바베큐", "바비큐"),
    ("쇼파", "소파"),
    ("비지니스", "비즈니스"),
    ("데이타", "데이터"),
    ("렌트카", "렌터카"),
    ("런닝", "러닝"),
    ("플랭카드", "플래카드"),
    ("워크샵", "워크숍"),
    ("앵콜", "앙코르"),
    ("심볼", "심벌"),
    ("로보트", "로봇"),
    ("타겟", "타깃"),
    ("팜플렛", "팸플릿"),
    ("자켓", "재킷"),
    ("쟈켓", "재킷"),
    ("알콜", "알코올"),
    ("스케쥴", "스케줄"),
    ("썬크림", "선크림"),
    ("써비스", "서비스"),
    ("라이센스", "라이선스"),
    ("엔돌핀", "엔도르핀"),
    ("메뉴얼", "매뉴얼"),
    ("메니저", "매니저"),
    ("매니아", "마니아"),
    ("플래쉬", "플래시"),
    ("프리젠테이션", "프레젠테이션"),
    ("디렉토리", "디렉터리"),
    ("컨셉", "콘셉트"),
    ("져지", "저지"),
    ("쉐이크", "셰이크"),
    ("쉐프", "셰프"),
    ("레포트", "리포트", WS),
    ("카라멜", "캐러멜"),
    ("샌달", "샌들"),
    ("부페", "뷔페"),
    ("돈까스", "돈가스"),
    ("나레이션", "내레이션"),
    ("뱃지", "배지"),
    ("앙케이트", "앙케트"),
    ("밧데리", "배터리"),
    ("밧테리", "배터리"),
]

# ---------------------------------------------------------------------------
# 문맥 의존 (대부분 오용이지만 정상 용법이 있음 → WARNING, 자동 교정 안 함)
# ---------------------------------------------------------------------------
_CONTEXT = [
    ("바램", "바람", {"word_start": True,
                     "message": "'소망'의 뜻이라면 '바람'이 맞습니다. ('색이 바래다'의 활용이면 정상)"}),
    ("설레임", "설렘", {"message": "명사형은 '설렘'이 맞습니다. (상품명이면 무시)"}),
    ("문안하게", "무난하게", {"message": "'무난하다'의 뜻이라면 '무난하게'가 맞습니다."}),
    ("구지", "굳이", {"word_start": True,
                    "message": "'고집스럽게'의 뜻이라면 '굳이'가 맞습니다. (지명·인명이면 무시)"}),
    ("일부로", "일부러", {"word_start": True,
                       "message": "'고의로'의 뜻이라면 '일부러'가 맞습니다. ('일부분으로'의 뜻이면 정상)"}),
    ("잊혀지", "잊히", {"message": "이중 피동입니다. '잊히다'를 권장합니다."}),
    ("되어지", "되", {"message": "이중 피동입니다. '되다'를 권장합니다."}),
    ("보여지", "보이", {"message": "이중 피동입니다. '보이다'를 권장합니다."}),
]


WORD_RULES = (
    word_rules("doe", _DOE)
    + word_rules("adv", _ADVERB)
    + word_rules("noun", _NOUN)
    + word_rules("loan", _LOANWORD, level=Level.WARNING, category=Category.STYLE)
    + word_rules("ctx", _CONTEXT, level=Level.WARNING)
)


# ============================================================================
# 어미 규칙
# ============================================================================

# 어절 끝 또는 '요'·문장부호가 뒤따를 때
_END = r"(?=요|[\s.,!?~\"')\]]|$)"


def _if_rieul(repl):
    """앞 음절 받침이 ㄹ일 때만 교정하는 fix 생성기."""
    return lambda m: repl(m) if jongseong(m.group(1)) == "ㄹ" else None


ENDING_RULES = [
    # -ㄹ께 → -ㄹ게   (할께요, 갈께)   '함께'는 받침 ㅁ이라 제외
    RegexRule("ending.001", r"([가-힣])께(?=라|란|요|[\s.,!?~\"')\]]|$)",
              _if_rieul(lambda m: m.group(1) + "게"),
              "'-ㄹ게'로 적습니다. (소리는 [께]로 나도 '-ㄹ게')"),

    # -ㄹ꺼(에요) → -ㄹ 거(예요)
    RegexRule("ending.002", r"([가-힣])꺼(에요)?(?=[야예요니면지라]|[\s.,!?~]|$)",
              _if_rieul(lambda m: m.group(1) + " 거" + ("예요" if m.group(2) else "")),
              "의존명사 '거'로 적고 띄어 씁니다. (할 거야, 할 거예요)"),

    # -ㄹ껄 → -ㄹ걸
    RegexRule("ending.003", r"([가-힣])껄" + _END,
              _if_rieul(lambda m: m.group(1) + "걸"),
              "'-ㄹ걸'로 적습니다."),

    # -ㄹ런지 → -ㄹ는지
    RegexRule("ending.004", r"([가-힣])런지",
              _if_rieul(lambda m: m.group(1) + "는지"),
              "'-ㄹ는지'가 맞습니다."),

    # -ㄹ려고/-ㄹ려면 (하다·보다 계열의 잘못된 ㄹ 첨가)
    RegexRule("ending.005", r"(할|볼|될)려(고|면|는|던)",
              lambda m: {"할": "하", "볼": "보", "될": "되"}[m.group(1)]
              + "려" + m.group(2),
              "'-려고'·'-려면' 앞에 ㄹ을 덧붙이지 않습니다. (하려고, 보려면)"),

    # -을려고 → -으려고  (먹을려고, 읽을려면)
    RegexRule("ending.006", r"을려(고|면|는|던)",
              lambda m: "으려" + m.group(1),
              "'-으려고'가 맞습니다."),

    # 있슴/없슴/했슴 → 있음/없음/했음  (뒤에 '니다'가 오면 습니다 규칙이 처리)
    RegexRule("ending.007", r"(있|없|했|겠|었|았|였)슴(?!니)",
              lambda m: m.group(1) + "음",
              "명사형 어미는 '-음'입니다. (있음, 했음)"),

    # 단독으로 쓰인 '않' + 공백 + 용언 → '안'  (않 된다, 않 해)
    RegexRule("ending.008", r"(?<![가-힣])않(?= [가-힣])",
              lambda m: "안",
              "용언 앞에서 부정 부사는 '안'입니다. ('않'은 '-지 않다'에만 씁니다)"),

    # -지 안았다/안습니다 → -지 않았다
    RegexRule("ending.009", r"지 안(았|습|는다|아요|아서|으면|는)",
              lambda m: "지 않" + m.group(1),
              "'-지 않다'로 적습니다."),

    # 단독 '왠' + 공백 → '웬'  ('왠'은 '왠지'에만 씁니다)
    RegexRule("ending.010", r"(?<![가-힣])왠(?= [가-힣])",
              lambda m: "웬",
              "'어찌 된'의 뜻이면 '웬'입니다. ('왠'은 '왠지'에만 씁니다)"),

    # 댓가 → 대가 (댓가지는 정상)
    RegexRule("ending.011", r"댓가(?!지)",
              lambda m: "대가",
              "'대가(代價)'가 맞습니다."),

]


# ============================================================================
# 띄어쓰기 규칙
# ============================================================================

def _dep(rule_id, noun, after, exceptions, message):
    """-ㄹ + 의존명사(noun) [+ after] 붙여쓰기 → 띄어쓰기."""
    def fix(m):
        if jongseong(m.group(1)) != "ㄹ":
            return None
        return f"{m.group(1)} {noun}{m.group(2)}"
    return RegexRule(rule_id, rf"([가-힣]){noun}({after})", fix, message,
                     category=Category.SPACING, exceptions=exceptions)


def _fix_su(m):
    pre, sp1, sp2, verb = m.group(1), m.group(2), m.group(3), m.group(4)
    if jongseong(pre) != "ㄹ" or (sp1 and sp2):
        return None
    return f"{pre} 수 {verb}"


SPACING_RULES = [
    # 할 수 있다/없다 (앞뒤 띄어쓰기 모두 점검)
    RegexRule("spacing.001", r"([가-힣])( ?)수( ?)(있|없)", _fix_su,
              "의존명사 '수'는 띄어 씁니다. (할 수 있다)",
              category=Category.SPACING),

    _dep("spacing.002", "것", "", ["날것", "물것", "들것", "별것", "탈것", "날것들"],
         "의존명사 '것'은 띄어 씁니다. (할 것)"),

    _dep("spacing.003", "때", "(?!까치)", ["물때", "썰물때", "밀물때", "끝물때"],
         "'때'는 앞말과 띄어 씁니다. (갈 때)"),

    _dep("spacing.004", "줄", "(?= ?(?:알|몰|모))", ["글줄", "날줄", "물줄", "밧줄"],
         "의존명사 '줄'은 띄어 씁니다. (할 줄 안다)"),

    # 부정 부사 '안' + 하다 활용형 → 띄어 씀  (안하기로 → 안 하기로)
    # 단어 첫머리의 '안'만 대상 ('불안하다', '편안하다', '미안하다'는 제외)
    # 뒤에 오는 활용형을 좁게 지정해 '안하늘', '안해숙' 같은 인명·'안할트' 같은 지명을 피한다
    RegexRule("spacing.005",
              r"(?<![가-힣])안(?=하(?:기|고|면|니|려|는|지|자|게)|해(?:도|서|요|야|[\s.,!?~]|$)"
              r"|했|할(?:게|래|거|까|[\s.,!?~]|$)|한(?:다|대|데)|합니다|함(?:[\s.,!?~]|$))",
              lambda m: "안 ",
              "부정의 '안'은 뒤의 '하다'와 띄어 씁니다. (안 하다)",
              category=Category.SPACING,
              exceptions=["안하무인"]),
]


# ============================================================================
# 검사 엔진
# ============================================================================

DEFAULT_RULES = [*WORD_RULES, *ENDING_RULES, *SPACING_RULES]


class Checker:
    """
    >>> Checker().check("오랫만에 만나서 반가웠어요. 내일 연락할께요.").checked
    '오랜만에 만나서 반가웠어요. 내일 연락할게요.'
    """

    def __init__(self, rules: Optional[Sequence[Rule]] = None,
                 ignore: Iterable[str] = (),
                 disabled: Iterable[str] = (),
                 categories: Optional[Iterable[Category]] = None):
        self.rules: List[Rule] = list(DEFAULT_RULES if rules is None else rules)
        self.ignore = set(ignore)            # 사용자 사전: 교정하지 않을 원문 표기
        self.disabled = set(disabled)        # 끌 규칙 ID
        self.categories = set(categories) if categories else None

    # --- 확장 API -------------------------------------------------------
    def add_rule(self, rule: Rule) -> "Checker":
        self.rules.append(rule)
        return self

    # --- 검사 ---------------------------------------------------------
    def find(self, text: str) -> List[Correction]:
        """겹침이 해소된 교정 목록 (원문 위치 순)."""
        found: List[Correction] = []
        for rule in self.rules:
            if rule.id in self.disabled:
                continue
            if self.categories and rule.category not in self.categories:
                continue
            for c in rule.find(text):
                if c.original in self.ignore:
                    continue
                found.append(c)
        return _resolve_overlaps(found)

    def check(self, text: str, apply_warnings: bool = False) -> Result:
        t0 = time.perf_counter()
        corrections = self.find(text)
        to_apply = [c for c in corrections
                    if apply_warnings or c.level is Level.ERROR]
        checked = _apply(text, to_apply)
        return Result(original=text, checked=checked, corrections=corrections,
                      time=time.perf_counter() - t0)


def _resolve_overlaps(cs: List[Correction]) -> List[Correction]:
    # 시작 위치 순, 같으면 더 긴 매치·ERROR 우선. 앞에서 채택된 범위와 겹치면 버림.
    cs.sort(key=lambda c: (c.start, -(c.end - c.start), c.level is not Level.ERROR))
    out: List[Correction] = []
    last_end = -1
    for c in cs:
        if c.start >= last_end:
            out.append(c)
            last_end = c.end
    return out


def _apply(text: str, cs: List[Correction]) -> str:
    parts, pos = [], 0
    for c in cs:  # 이미 위치 순·비겹침
        parts.append(text[pos:c.start])
        parts.append(c.suggestion)
        pos = c.end
    parts.append(text[pos:])
    return "".join(parts)


_default: Optional[Checker] = None


def check(text: str, apply_warnings: bool = False) -> Result:
    """기본 규칙으로 검사하는 단축 함수."""
    global _default
    if _default is None:
        _default = Checker()
    return _default.check(text, apply_warnings=apply_warnings)


# ============================================================================
# py-hanspell 호환 계층
# ============================================================================
#
# py-hanspell 호환 계층.
#
# 기존 코드의 import 한 줄만 바꾸면 네이버 의존 없이 동작하도록 하는 것이 목적.
#
#     # from hanspell import spell_checker
#     from matchum.compat.hanspell import spell_checker
#
#     result = spell_checker.check("안녕 하세요. 몇일 뒤에 봐요.")
#     result.checked
#     result.as_dict()

class CheckResult:
    PASSED = 0
    WRONG_SPELLING = 1
    WRONG_SPACING = 2
    AMBIGUOUS = 3
    STATISTICAL_CORRECTION = 4


class Checked:
    def __init__(self, result=False, original="", checked="", errors=0, words=None, time=0.0):
        self.result = result
        self.original = original
        self.checked = checked
        self.errors = errors
        self.words = words or OrderedDict()
        self.time = time

    def as_dict(self):
        return {
            "result": self.result, "original": self.original, "checked": self.checked,
            "errors": self.errors, "words": self.words, "time": self.time,
        }

    def only_checked(self):
        return self.checked


def _word_status(r) -> "OrderedDict[str, int]":
    # 교정문 어절별 상태 (py-hanspell 형식 근사)
    changed_cat = {}
    for c in r.corrections:
        changed_cat[c.suggestion.strip()] = (
            CheckResult.WRONG_SPACING if c.category is Category.SPACING
            else CheckResult.WRONG_SPELLING
        )
    words = OrderedDict()
    for w in r.checked.split():
        status = CheckResult.PASSED
        for frag, cat in changed_cat.items():
            if frag and frag.split()[0] in w:
                status = cat
                break
        words[w] = status
    return words


def _hanspell_check(sentences: Union[str, List[str]]):
    if isinstance(sentences, list):
        return [_hanspell_check(s) for s in sentences]
    r = check(sentences)
    return Checked(result=True, original=r.original, checked=r.checked,
                   errors=r.errors, words=_word_status(r), time=r.time)


spell_checker = SimpleNamespace(check=_hanspell_check)


# ============================================================================
# 명령줄 도구
# ============================================================================
#
# 명령줄 도구.
#
#     matchum "몇일 뒤에 연락할께요"
#     matchum -f 원고.txt --json
#     echo "할수있다" | matchum

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="matchum", description="오프라인 한국어 맞춤법 검사")
    p.add_argument("text", nargs="?", help="검사할 문장 (생략 시 표준입력)")
    p.add_argument("-f", "--file", help="검사할 텍스트 파일 (UTF-8)")
    p.add_argument("--json", action="store_true", help="JSON으로 출력")
    p.add_argument("--apply-warnings", action="store_true", help="WARNING 수준도 교정문에 반영")
    args = p.parse_args(argv)

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    r = Checker().check(text, apply_warnings=args.apply_warnings)

    if args.json:
        print(json.dumps(r.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(r.checked)
        for c in r.corrections:
            mark = "!" if c.level.value == "error" else "?"
            print(f"  {mark} {c.original} → {c.suggestion}  ({c.message}) [{c.rule_id}]",
                  file=sys.stderr)
    return 1 if r.corrections else 0



__version__ = "0.1.3"
__all__ = ["Checker", "check", "Rule", "RegexRule", "WordRule", "word_rules",
           "Category", "Correction", "Level", "Result", "spell_checker", "CheckResult",
           "DEFAULT_RULES"]


if __name__ == "__main__":
    raise SystemExit(main())

