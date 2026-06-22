#!/usr/bin/env python3
"""
evaluate_kexaone.py
EXAONE-3.5-32B 모델 전체 시스템 평가 스크립트 (Solar 테스트와 동일 조건)
포트 8003에서 서빙 중인 exaone-32b 모델 사용

사용법:
  DISPLAY=:0 python evaluate_kexaone.py --quick
  DISPLAY=:0 python evaluate_kexaone.py --system 도서관
  nohup bash -c 'DISPLAY=:0 python evaluate_kexaone.py' > eval_exaone32b_full.log 2>&1 &
"""

import asyncio
import csv
import json
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from browser_use import BrowserSession, BrowserProfile
from browser_use.llm.openai.chat import ChatOpenAI
from logged_agent import LoggedAgent

load_dotenv()

# ============================================================
# 설정
# ============================================================
VLLM_MODEL = "exaone-32b"
VLLM_BASE_URL = "http://localhost:8003/v1"
RESULT_CSV = f"./results_exaone32b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
MAX_STEPS = 20
SLEEP_BETWEEN_TASKS = 3

CNU_ID = os.getenv("CNU_ID", "")
CNU_PW = os.getenv("CNU_PASSWORD", "")
KORUS_ID = os.getenv("KORUS_ID", "")
KORUS_PW = os.getenv("KORUS_PASSWORD", "")

# Solar 테스트 시스템 메시지
EXTEND_SYSTEM_MESSAGE = (
    "You are an expert web agent navigating Chungnam National University services.\n"
    "ALL final answers in the `done` action MUST be in Korean.\n"
    "If an alert or popup appears, read the message and call done immediately.\n"
)

# ============================================================
# 시스템별 설정 (Solar 테스트와 동일한 URL 및 base_context)
# ============================================================
SYSTEM_CONFIG = {
    "통합정보시스템": {
        "start_url": "https://portal.cnu.ac.kr",
        "base_context": (
            f"포털(portal.cnu.ac.kr) 로그인 페이지입니다. "
            f"Step 1: 아이디 입력란에 {CNU_ID} 입력. "
            f"Step 2: 비밀번호 입력란에 {CNU_PW} 입력. "
            f"Step 3: 로그인 버튼 클릭. "
            f"Step 4: 반드시 wait action으로 15초 대기. "
            f"Step 5: 15초 후 URL 확인. /proc/Main이면 성공. '통합정보시스템' 버튼 클릭 후 새 탭 전환. "
            f"URL이 /proc/Login.eps이면 Step 1부터 다시. navigate는 절대 하지 말 것.\n"
        ),
        "sensitive_data": None,
    },
    "도서관": {
        "start_url": "https://library.cnu.ac.kr/",
        "base_context": f"충남대학교 도서관 사이트에 접속되어 있습니다. 로그인이 필요할 때만 아이디 {CNU_ID}, 비밀번호 {CNU_PW}로 로그인하세요.\n",
        "sensitive_data": None,
    },
    "학사지원시스템": {
        "start_url": "https://with.cnu.ac.kr/",
        "base_context": f"CNU With U+ 서비스에 접속되어 있습니다. 로그인이 필요하면 아이디 {CNU_ID}, 비밀번호 {CNU_PW}로 로그인하세요.\n",
        "sensitive_data": None,
    },
    "사이버캠퍼스": {
        "start_url": "https://dcs-lcms.cnu.ac.kr/",
        "base_context": f"사이버캠퍼스에 접속되어 있습니다. 로그인이 필요하면 아이디 {CNU_ID}, 비밀번호 {CNU_PW}로 로그인하세요.\n",
        "sensitive_data": None,
    },
    "전자결재": {
        "start_url": "https://cnu.korus.ac.kr/",
        "base_context": f"충남대학교 전자결재 시스템에 접속되어 있습니다. 로그인이 필요하면 아이디 {KORUS_ID}, 비밀번호 {KORUS_PW}로 로그인하세요.\n",
        "sensitive_data": None,
    },
    "학과홈페이지": {
        "start_url": "https://computer.cnu.ac.kr/computer/index.do",
        "base_context": "학과 사이트에 접속되어 있습니다.\n",
        "sensitive_data": None,
    },
    "통합": {
        "start_url": "https://portal.cnu.ac.kr",
        "base_context": f"충남대학교 여러 시스템을 연계합니다. 아이디 {CNU_ID}, 비밀번호 {CNU_PW}로 로그인하세요.\n",
        "sensitive_data": None,
    },
}

# ============================================================
# 전체 태스크
# ============================================================
TASKS = {
    "학과홈페이지": [
        {"id": 1,  "task": "컴퓨터융합학부 학부 소개 페이지에서 학부장이 누구야?"},
        {"id": 2,  "task": "컴퓨터융합학부 교수님 목록 알려줘"},
        {"id": 3,  "task": "컴퓨터융합학부 행정직원(조교) 정보 알려줘"},
        {"id": 4,  "task": "컴퓨터융합학부 이영석 교수님 이메일 주소 알려줘"},
        {"id": 5,  "task": "컴퓨터융합학부 이영석 교수님 연구 분야가 뭐야?"},
        {"id": 6,  "task": "컴퓨터융합학부 2021년 및 2022년 2월 졸업생들의 대학원 진학률 알려줘"},
        {"id": 7,  "task": "컴퓨터융합학부 2021년 및 2022년 2월 졸업생들의 취업률 알려줘"},
        {"id": 8,  "task": "컴퓨터융합학부 사무실 위치 어디야"},
        {"id": 9,  "task": "컴퓨터융합학부 전공기초 과목 알려줘"},
        {"id": 10, "task": "컴퓨터융합학부 가장 최신 학사공지 알려줘"},
        {"id": 11, "task": "컴퓨터융합학부 수강신청 관련 학사공지 알려줘"},
        {"id": 12, "task": "컴퓨터융합학부 계절학기 관련 공지 알려줘"},
        {"id": 13, "task": "컴퓨터융합학부 졸업 관련 학사공지 알려줘"},
        {"id": 14, "task": "컴퓨터융합학부 장학금 관련 학사공지 있어?"},
        {"id": 15, "task": "컴퓨터융합학부 교내 일반 소식 가장 최신 것 알려줘"},
        {"id": 16, "task": "컴퓨터융합학부 SW/AI 관련 교내 소식 있어?"},
        {"id": 17, "task": "컴퓨터융합학부 Project Fair 접수 안내 공지 알려줘"},
        {"id": 18, "task": "컴퓨터융합학부 코딩 테스트 관련 교내 공지 있어?"},
        {"id": 19, "task": "컴퓨터융합학부 가장 최신 취업·인턴 공지 알려줘"},
        {"id": 20, "task": "컴퓨터융합학부 소프트웨어중심대학 관련 사업단 소식 알려줘"},
    ],
    "도서관": [
        {"id": 1,  "task": "'인공지능'으로 검색해서 대출 가능한 책 목록을 알려줘."},
        {"id": 2,  "task": "'머신러닝'으로 검색해서 책 제목과 저자를 알려줘."},
        {"id": 3,  "task": "'딥러닝'으로 검색해서 소장 중인 책 목록을 알려줘."},
        {"id": 4,  "task": "'알고리즘'으로 검색해서 대출 가능한 책을 알려줘."},
        {"id": 5,  "task": "저자 '남궁성'으로 검색해서 소장 중인 책 목록을 알려줘."},
        {"id": 6,  "task": "저자 '윤성우'로 검색해서 소장 중인 책을 알려줘."},
        {"id": 7,  "task": "저자 '홍정모'로 검색해서 소장 중인 책을 알려줘."},
        {"id": 8,  "task": "전자자료 메뉴에서 '인공지능'을 검색해서 결과를 알려줘."},
        {"id": 9,  "task": "'머신러닝' 관련 전자자료를 검색해줘."},
        {"id": 10, "task": "'자바의 정석' 소장 위치와 청구기호를 알려줘."},
        {"id": 11, "task": "'인공지능' 소장 위치와 대출 가능 여부를 알려줘."},
        {"id": 12, "task": "'컴퓨터 구조' 소장 위치와 청구기호를 알려줘."},
        {"id": 13, "task": "도서 대출 기간과 권수를 알려줘."},
        {"id": 14, "task": "도서 연장 방법을 알려줘."},
        {"id": 15, "task": "도서 반납하려면 어디로 몇 시에 가야 하는지 알려줘."},
        {"id": 16, "task": "상호대차 서비스 신청 방법을 알려줘."},
        {"id": 17, "task": "현재 대출 중인 도서 목록을 알려줘."},
        {"id": 18, "task": "반납 기한이 임박한 도서가 있는지 알려줘."},
        {"id": 19, "task": "대출 중인 도서의 반납 기한을 알려줘."},
        {"id": 20, "task": "희망도서 신청 내역을 알려줘."},
    ],
    "통합정보시스템": [
        {"id": 0,  "task": "미래설계상담 신청해줘"},
        {"id": 1,  "task": "현재 수강중인 종합설계 수강 인원 알려줘"},
        {"id": 2,  "task": "이번 학기 취업과 창업 들으려고 하는데 안기돈 교수님 수업으로 리스트 뽑아줘"},
        {"id": 3,  "task": "이번학기 수강하는 과목들의 강의실 위치 정리해줘."},
        {"id": 4,  "task": "나 지금 졸업하려면 어떤거 더 채워야해?"},
        {"id": 5,  "task": "전체 평점 4.0을 넘으려면 계획을 어떻게 세우면 될까?"},
        {"id": 6,  "task": "이번주 이영석 교수님과 미팅을 해야하는데 교수님 시간표 보고 되는 시간으로 알려줘."},
        {"id": 7,  "task": "공5호관 415 강의실 사용 가능한 시간들이 언제인지 알려줘"},
        {"id": 8,  "task": "지금까지 이수한 총 학점이 몇점이고, 얼마나 남았는지 알려줘."},
        {"id": 9,  "task": "저번주에 나 컴퓨터프로그래밍3 과목을 결석했는데 출석 인정 처리 해줘."},
        {"id": 10, "task": "재이수 가능한 과목 리스트만 뽑아서 재수강 계획 세워줘."},
        {"id": 11, "task": "현재 학점이랑 백분위 알려줘."},
        {"id": 12, "task": "제주대학교로 학점교류 신청해줘. 6/30~7/21이고, 수강 과목 이름은 '자바의 기초', 00분반이야. 일반 선택으로 신청하면 돼. 다 하고 신청서 출력해줘."},
        {"id": 13, "task": "사회봉사활동 중에서 신청 가능한 것들만 리스트 뽑아줘."},
        {"id": 14, "task": "등록금 고지서 출력해줘."},
        {"id": 15, "task": "나 이번학기에 장학금 받은거 있으면 알려줘."},
        {"id": 16, "task": "백마인턴십 신청 내역 확인해줘."},
        {"id": 17, "task": "나에게 맞는 백마인턴십 추천해줘."},
        {"id": 18, "task": "전공 심화 리스트만 뽑아줘."},
        {"id": 19, "task": "핵심 교양 리스트만 뽑아줘."},
        {"id": 20, "task": "졸업자가진단 내용 정리해서 알려줘."},
        {"id": 21, "task": "운영체제 과목의 강의계획서 확인하고 나에게 정리해서 알려줘."},
        {"id": 22, "task": "졸업자가진단에서 불합격인것들 정리해서 알려줘."},
    ],
    "학사지원시스템": [
        {"id": 1,  "task": "2026년 1학기 개인 비교과 프로그램 최신 등록된 3개 알려줘"},
        {"id": 2,  "task": "2026년 2학기 개인 비교과 프로그램 뭐 있는지 알려줘"},
        {"id": 3,  "task": "2026년 1학기 개인 비교과 프로그램 중 프로그램 유형 강의형 뭐 있는 지알려줘"},
        {"id": 4,  "task": "2026년 1학기 개인 비교과 프로그램 중 프로그램 유형 참여형 뭐 있는 지알려줘"},
        {"id": 5,  "task": "2026년 1학기 개인 비교과 프로그램 중 프로그램 유형 파견형 뭐 있는 지알려줘"},
        {"id": 6,  "task": "2026년 1학기 개인 비교과 프로그램 중 프로그램 유형 상담형 뭐 있는 지알려줘"},
        {"id": 7,  "task": "2026년 1학기 개인 비교과 프로그램 중 프로그램 유형 실습형 뭐 있는 지알려줘"},
        {"id": 8,  "task": "2026년 1학기 개인 비교과 프로그램 중 프로그램 유형 사회형 뭐 있는 지알려줘"},
        {"id": 9,  "task": "2026년 1학기 개인 비교과 프로그램 중 1주일 내 등록된 비교과 프로그램 알려줘"},
        {"id": 10, "task": "2026년 1학기 개인 비교과 프로그램 중 1주일 내 마감인 비교과 프로그램 알려줘"},
        {"id": 11, "task": "2026년 1학기 개인 비교과 프로그램 중 신청 마감 제일 임박한 비교과 프로그램 알려줘"},
        {"id": 12, "task": "나의 개인 비교과 프로그램 신청 내역 알려줘"},
        {"id": 13, "task": "2025년 나의 개인비교과 수료 프로그램 수 알려줘"},
        {"id": 14, "task": "2025년 나의 개인비교과 총 인정시간 알려줘"},
        {"id": 15, "task": "2026년 나의 개인비교과 수료 프로그램 수 알려줘"},
        {"id": 16, "task": "2026년 나의 개인비교과 총 인정시간 알려줘"},
        {"id": 17, "task": "지금 신청할 수 있는 그룹비교과 프로그램 알려줘"},
        {"id": 18, "task": "STRONG+ACTIVATE 역량이 뭔지 알려줘"},
        {"id": 19, "task": "자기관리 역량이 뭔지 알려줘"},
        {"id": 20, "task": "의사소통 역량이 뭔지 알려줘"},
    ],
    "사이버캠퍼스": [
        {"id": 1,  "task": "강의실에서 모든 수강 과목명을 알려줘"},
        {"id": 2,  "task": "컴퓨터프로그래밍3 과목 페이지로 이동해줘"},
        {"id": 3,  "task": "컴퓨터프로그래밍3 과목 공지사항에서 가장 최근 글 제목을 알려줘"},
        {"id": 4,  "task": "범죄의진실과오해 과목의 공지사항 목록을 보여줘"},
        {"id": 5,  "task": "컴퓨터프로그래밍3 과목 자료실에서 가장 최근 파일 이름을 알려줘"},
        {"id": 6,  "task": "컴퓨터프로그래밍3 과목에 제출해야 할 과제가 있는지 확인해줘"},
        {"id": 7,  "task": "컴퓨터프로그래밍3 과목 과제 목록에서 가장 최근 과제 제목과 마감일을 알려줘"},
        {"id": 8,  "task": "컴퓨터프로그래밍3 과목 강의수강 페이지에서 이번 주차 강의 제목을 알려줘"},
        {"id": 9,  "task": "종합설계1 과목의 출석 현황을 확인해줘"},
        {"id": 10, "task": "컴퓨터프로그래밍3 과목 성적 페이지에서 현재까지 받은 점수를 확인해줘"},
        {"id": 11, "task": "컴퓨터프로그래밍3 과목 자유게시판에 글이 있는지 확인해줘"},
        {"id": 12, "task": "종합설계1 과목의 팀프로젝트 페이지에 등록된 프로젝트가 있는지 확인해줘"},
        {"id": 13, "task": "받은쪽지함에 새로운 쪽지가 있는지 확인해줘"},
        {"id": 14, "task": "To-Do-list에 등록된 할 일 목록을 알려줘"},
        {"id": 15, "task": "컴퓨터프로그래밍3 과목 Q&A 게시판에 '중간고사 범위 질문'이라는 제목으로 '중간고사 범위가 어디까지인가요?'라는 내용의 글을 작성해줘"},
        {"id": 16, "task": "종합설계1 과목 자유게시판에 '회의록 공유'라는 제목으로 '4주차 팀 회의록 공유합니다'라는 글을 작성해줘"},
        {"id": 17, "task": "종합설계1 과목 Q&A에 'GitHub 레포 관련'이라는 제목으로 '팀 GitHub 레포지토리 링크를 어디에 제출하면 되나요?'라는 글을 작성해줘"},
        {"id": 18, "task": "컴퓨터프로그래밍3 과목과 종합설계1 과목의 과제 페이지를 각각 확인해서 미제출 과제가 있는 과목을 알려줘"},
        {"id": 19, "task": "메인 페이지에서 To-Do-list와 알림마당을 순서대로 확인하고 오늘 해야 할 일이 있는지 알려줘"},
        {"id": 20, "task": "컴퓨터프로그래밍3 과목 강의수강 페이지에서 이번 주차 강의를 확인하고, 해당 주차에 과제가 있는지 과제 페이지에서도 확인해줘"},
    ],
    "전자결재": [
        {"id": 1,  "task": "비전자 일괄결재 검토"},
        {"id": 2,  "task": "결재대기함 비전자문서 일괄로 검토 부탁해"},
        {"id": 3,  "task": "결재함 확인해줘"},
        {"id": 4,  "task": "대기 중인 결재 문서 검토해줘"},
        {"id": 5,  "task": "전자문서 결재대기함에서 받은 결재 건들 본문이랑 첨부파일 확인 부탁해"},
        {"id": 6,  "task": "결재 문서 미리 확인"},
        {"id": 7,  "task": "대기 중인 결재 문서 내용 요약해줘"},
        {"id": 8,  "task": "이수증 등록해줘"},
        {"id": 9,  "task": "출장 신청 좀 해줘"},
        {"id": 10, "task": "받은 공문 첨부 일괄 다운"},
        {"id": 11, "task": "받은 공문 확인해줘"},
        {"id": 12, "task": "수신 공문함 열어서 보여줘"},
        {"id": 13, "task": "오늘 들어온 공문 리스트 보여주세요"},
        {"id": 14, "task": "전자문서 받은문서함에 새 공문 들어온 거 있나 확인하고 싶은데, 제목이랑 발신처 날짜 다 보여줘"},
        {"id": 15, "task": "출장 문서 찾아줘"},
        {"id": 16, "task": "진행문서함에서 연구비 신청 문서 검색해줘"},
        {"id": 17, "task": "이번 달에 기안된 장비 구매 관련 문서 찾아주세요"},
        {"id": 18, "task": "완료문서함에서 '학생 지도' 키워드로 제목 검색하고, 2026년 4월 이후 기안된 건만 보여줘"},
        {"id": 19, "task": "완료 문서 보여줘"},
        {"id": 20, "task": "결재 완료된 문서들 목록 좀 확인하고 싶어"},
    ],
    "통합": [
        {"id": 1, "task": "포털에서 오늘 내가 듣는 수업 확인해주고 사이버캠퍼스에서 그 수업 최근 공지사항 뭐 올라왔는지 확인해줘"},
        {"id": 2, "task": "포털에서 신착도서 확인해주고 가장 앞에 있는 도서를 도서관에서 검색해줘"},
        {"id": 3, "task": "통합정보시스템 수강정보에서 26년 1학기 과목 확인해주고, AI 관련된 과목이 있으면 그 과목의 강의계획서를 조회하고, 사용 교재의 참고 문헌이 있으면 그 교재 도서관에서 검색해줘"},
        {"id": 4, "task": "통합정보시스템에서 졸업자가진단 들어가주고 전공 분야에서 C+ 이하의 학점을 맞은 과목이 있으면 그 과목의 강의계획서를 조회해줘. 그리고 사용 교재의 참고 문헌이 있으면 그 교재를 도서관에서 검색해줘"},
    ],
}


def get_llm():
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="dummy",
        temperature=0.0,
        max_completion_tokens=4096,
        timeout=100.0,
    )


def parse_trajectory(task: str, system: str, history, total_elapsed: float) -> dict:
    steps = []
    for step_idx, h in enumerate(history.history):
        step_duration = None
        if h.metadata and hasattr(h.metadata, "duration_seconds"):
            step_duration = round(h.metadata.duration_seconds, 2)
        step_data = {
            "step": step_idx + 1,
            "duration_seconds": step_duration,
            "evaluation": None,
            "memory": None,
            "next_goal": None,
            "actions": [],
            "results": [],
        }
        if h.model_output:
            step_data["evaluation"] = h.model_output.evaluation_previous_goal
            step_data["memory"] = h.model_output.memory
            step_data["next_goal"] = h.model_output.next_goal
            for action in h.model_output.action:
                action_dict = action.model_dump(exclude_none=True)
                for key, val in action_dict.items():
                    if val is not None:
                        step_data["actions"].append({"type": key, "params": val})
                        break
        for result in h.result:
            step_data["results"].append({
                "is_done": result.is_done,
                "success": result.success,
                "error": result.error,
                "memory": result.long_term_memory,
            })
        steps.append(step_data)

    final_result = None
    success = False
    for h in reversed(history.history):
        if h.model_output:
            for action in h.model_output.action:
                action_dict = action.model_dump(exclude_none=True)
                if "done" in action_dict:
                    final_result = action_dict["done"].get("text")
                    success = action_dict["done"].get("success", False)
                    break
        if final_result:
            break

    history_duration = round(history.total_duration_seconds(), 2) if hasattr(history, "total_duration_seconds") else None

    return {
        "task": task,
        "system": system,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "total_steps": len(steps),
        "total_duration_seconds": history_duration or round(total_elapsed, 2),
        "steps": steps,
        "final_result": final_result,
    }


def save_result(result: dict, csv_path: str):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def print_summary(results: list):
    from collections import defaultdict
    stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in results:
        s = r["system"]
        stats[s]["total"] += 1
        if r["success"]:
            stats[s]["success"] += 1

    print("\n" + "="*50)
    print("📊 EXAONE-3.5-32B 테스트 결과 요약")
    print("="*50)
    total_all = sum_success = 0
    for system, stat in stats.items():
        rate = stat["success"] / stat["total"] * 100
        print(f"  {system}: {stat['success']}/{stat['total']} ({rate:.1f}%)")
        total_all += stat["total"]
        sum_success += stat["success"]
    if total_all > 0:
        print(f"\n  전체: {sum_success}/{total_all} ({sum_success/total_all*100:.1f}%)")
    print("="*50)


async def run_task(system: str, task_info: dict, llm, task_idx: int, total: int) -> dict:
    task_id = task_info["id"]
    task = task_info["task"]
    config = SYSTEM_CONFIG[system]
    full_task = f"{config['base_context']}요청사항: {task}"

    print(f"\n[{task_idx}/{total}] [{system}] Task {task_id}: {task[:50]}...")

    USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
    
    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            viewport={"width": 1920, "height": 1080},
            proxy={"server": "socks5://localhost:8088"} if USE_PROXY else None,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        ),
        keep_alive=False,
    )
    await browser_session.start()

    # start_url로 먼저 이동
    if config.get("start_url"):
        try:
            await browser_session.navigate_to(config["start_url"])
        except Exception:
            pass
        await asyncio.sleep(8)

    # 통합정보시스템은 코드로 직접 로그인
    if system == "통합정보시스템":
        try:
            await browser_session.execute_javascript(f"""
                document.getElementById('user_id').value = '{CNU_ID}';
                document.getElementById('user_password').value = '{CNU_PW}';
                document.querySelector('button[type=submit], input[type=submit], .btn_login').click();
            """)
            await asyncio.sleep(10)
        except Exception as e:
            print(f"  로그인 시도 실패: {e}")
            
    start_time = time.time()
    steps = 0
    success = False
    response_text = ""

    try:
        agent = LoggedAgent(
            task=full_task,
            llm=llm,
            use_vision=False,
            validate_output=False,
            extend_system_message=EXTEND_SYSTEM_MESSAGE,
            calculate_cost=False,
            log_path=f"./data/training_data_exaone32b.jsonl",
            save_conversation_path=f"./data/conversations_exaone32b/task_{system}_{task_id:03d}.json",
            browser=browser_session,
        )

        history = await agent.run(max_steps=MAX_STEPS)

        elapsed = time.time() - start_time
        trajectory = parse_trajectory(task, system, history, elapsed)
        traj_path = Path(f"./trajectories_exaone32b/task_{system}_{task_id:03d}.json")
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        steps = trajectory["total_steps"]
        success = trajectory["success"]
        response_text = trajectory["final_result"] or "No result"

    except Exception as e:
        response_text = f"ERROR: {str(e)}"
        success = False
        print(f"  ❌ 오류: {e}")
    finally:
        elapsed = time.time() - start_time
        try:
            await browser_session.stop()
        except:
            pass

    result = {
        "system": system,
        "task_id": task_id,
        "task": task,
        "success": success,
        "response": response_text[:500],
        "steps": steps,
        "elapsed_sec": round(elapsed, 2),
        "steps_per_sec": round(elapsed / steps, 2) if steps > 0 else 0,
    }

    status = "✅" if success else "❌"
    print(f"  {status} {elapsed:.1f}초, {steps}스텝")
    return result


async def main():
    parser = argparse.ArgumentParser(description="EXAONE-3.5-32B 전체 시스템 평가")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    systems_to_test = [args.system] if args.system else list(TASKS.keys())
    max_per_system = 2 if args.quick else args.max_tasks

    os.makedirs("./data/conversations_exaone32b", exist_ok=True)

    llm = get_llm()
    all_results = []

    total_tasks = sum(
        len(TASKS[s][:max_per_system] if max_per_system else TASKS[s])
        for s in systems_to_test
    )

    print(f"🚀 EXAONE-3.5-32B 평가 시작 (Solar 동일 조건)")
    print(f"   모델: {VLLM_MODEL} @ {VLLM_BASE_URL}")
    print(f"   결과 저장: {RESULT_CSV}")
    print(f"   총 태스크: {total_tasks}개")

    task_idx = 0
    for system in systems_to_test:
        tasks = TASKS[system]
        if max_per_system:
            tasks = tasks[:max_per_system]

        print(f"\n{'='*50}")
        print(f"📌 {system} ({len(tasks)}개 태스크)")
        print(f"{'='*50}")

        for task_info in tasks:
            task_idx += 1
            result = await run_task(system, task_info, llm, task_idx, total_tasks)
            all_results.append(result)
            save_result(result, RESULT_CSV)
            await asyncio.sleep(SLEEP_BETWEEN_TASKS)

    print_summary(all_results)
    print(f"\n✅ 결과 저장 완료: {RESULT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())