#!/usr/bin/env python3
"""
collect_integrated.py
통합 시나리오 4개 테스트 (Gemini 2.5 Flash / GPT-4o-mini 선택)

사용법:
  DISPLAY=:0 nohup python collect_integrated.py --model gemini > logs_integrated_gemini.txt 2>&1 &
  DISPLAY=:0 nohup python collect_integrated.py --model gpt > logs_integrated_gpt.txt 2>&1 &
"""

import asyncio
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from browser_use import BrowserSession, BrowserProfile
from logged_agent_vl import LoggedAgentVL

load_dotenv()

MAX_STEPS = 50
SLEEP_BETWEEN_TASKS = 5

CNU_ID = os.getenv("CNU_ID", "")
CNU_PW = os.getenv("CNU_PASSWORD", "")

EXTEND_SYSTEM_MESSAGE = (
    "You are an expert web agent navigating Chungnam National University services.\n"
    "ALL final answers in the `done` action MUST be in Korean.\n"
    "If an alert or popup appears, read the message and call done immediately.\n"
    "This task may require navigating across multiple websites. Complete all steps before calling done.\n"
)

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
    },
    "도서관": {
        "start_url": "https://library.cnu.ac.kr/",
        "base_context": f"충남대학교 도서관 사이트에 접속되어 있습니다. 로그인이 필요할 때만 아이디 {CNU_ID}, 비밀번호 {CNU_PW}로 로그인하세요.\n",
    },
    "사이버캠퍼스": {
        "start_url": "https://dcs-lcms.cnu.ac.kr/",
        "base_context": f"사이버캠퍼스에 접속되어 있습니다. 로그인이 필요하면 아이디 {CNU_ID}, 비밀번호 {CNU_PW}로 로그인하세요.\n",
    },
    "포털": {
        "start_url": "https://portal.cnu.ac.kr",
        "base_context": (
            f"충남대학교 포털(portal.cnu.ac.kr)입니다. "
            f"아이디 {CNU_ID}, 비밀번호 {CNU_PW}로 로그인하세요.\n"
        ),
    },
}

# 통합 시나리오 4개
INTEGRATED_TASKS = [
    {
        "id": 1,
        "task": "포털에서 오늘 내가 듣는 수업 확인해주고 사이버캠퍼스에서 그 수업 최근 공지사항 뭐 올라왔는지 확인해줘",
        "start_url": "https://portal.cnu.ac.kr",
        "allowed_domains": ["portal.cnu.ac.kr", "dcs-lcms.cnu.ac.kr"],
        "base_context": (
            f"충남대학교 포털(portal.cnu.ac.kr)과 사이버캠퍼스(dcs-lcms.cnu.ac.kr)를 연계하는 작업입니다. "
            f"아이디 {CNU_ID}, 비밀번호 {CNU_PW}로 로그인하세요.\n"
            f"요청사항: "
        ),
    },
    {
        "id": 2,
        "task": "포털에서 신착도서 확인해주고 가장 앞에 있는 도서를 도서관에서 검색해줘",
        "start_url": "https://portal.cnu.ac.kr",
        "allowed_domains": ["portal.cnu.ac.kr", "library.cnu.ac.kr"],
        "base_context": (
            f"충남대학교 포털(portal.cnu.ac.kr)과 도서관(library.cnu.ac.kr)을 연계하는 작업입니다. "
            f"포털 로그인이 필요하면 아이디 {CNU_ID}, 비밀번호 {CNU_PW}로 로그인하세요.\n"
            f"요청사항: "
        ),
    },
    {
        "id": 3,
        "task": "통합정보시스템 수강정보에서 26년 1학기 과목 확인해주고, AI 관련된 과목이 있으면 그 과목의 강의계획서를 조회하고, 사용 교재의 참고 문헌이 있으면 그 교재 도서관에서 검색해줘",
        "start_url": "https://portal.cnu.ac.kr",
        "allowed_domains": ["portal.cnu.ac.kr", "library.cnu.ac.kr"],
        "base_context": (
            f"포털(portal.cnu.ac.kr) 로그인 페이지입니다. "
            f"Step 1: 아이디 입력란에 {CNU_ID} 입력. "
            f"Step 2: 비밀번호 입력란에 {CNU_PW} 입력. "
            f"Step 3: 로그인 버튼 클릭. "
            f"Step 4: wait action으로 15초 대기. "
            f"Step 5: '통합정보시스템' 버튼 클릭 후 새 탭 전환. "
            f"이후 도서관(library.cnu.ac.kr) 검색도 수행하세요.\n"
            f"요청사항: "
        ),
    },
    {
        "id": 4,
        "task": "통합정보시스템에서 졸업자가진단 들어가주고 전공 분야에서 C+ 이하의 학점을 맞은 과목이 있으면 그 과목의 강의계획서를 조회해줘. 그리고 사용 교재의 참고 문헌이 있으면 그 교재를 도서관에서 검색해줘",
        "start_url": "https://portal.cnu.ac.kr",
        "allowed_domains": ["portal.cnu.ac.kr", "library.cnu.ac.kr"],
        "base_context": (
            f"포털(portal.cnu.ac.kr) 로그인 페이지입니다. "
            f"Step 1: 아이디 입력란에 {CNU_ID} 입력. "
            f"Step 2: 비밀번호 입력란에 {CNU_PW} 입력. "
            f"Step 3: 로그인 버튼 클릭. "
            f"Step 4: wait action으로 15초 대기. "
            f"Step 5: '통합정보시스템' 버튼 클릭 후 새 탭 전환. "
            f"이후 도서관(library.cnu.ac.kr) 검색도 수행하세요.\n"
            f"요청사항: "
        ),
    },
]


def parse_trajectory(task: str, history, total_elapsed: float) -> dict:
    steps = []
    for step_idx, h in enumerate(history.history):
        step_duration = None
        if h.metadata and hasattr(h.metadata, "duration_seconds"):
            step_duration = round(h.metadata.duration_seconds, 2)
        step_data = {
            "step": step_idx + 1,
            "duration_seconds": step_duration,
            "evaluation": None, "memory": None, "next_goal": None,
            "actions": [], "results": [],
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
                "is_done": result.is_done, "success": result.success,
                "error": result.error, "memory": result.long_term_memory,
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
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "total_steps": len(steps),
        "total_duration_seconds": history_duration or round(total_elapsed, 2),
        "steps": steps,
        "final_result": final_result,
    }


async def run_task(task_info: dict, llm, model_name: str, task_idx: int, total: int) -> dict:
    task_id = task_info["id"]
    task = task_info["task"]
    full_task = task_info["base_context"] + task

    print(f"\n{'='*60}")
    print(f"[{task_idx}/{total}] [통합] Task {task_id}: {task[:60]}...")
    print('='*60)

    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            viewport={"width": 1920, "height": 1080},
            allowed_domains=task_info["allowed_domains"],
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        ),
        keep_alive=False,
    )
    await browser_session.start()

    # start_url로 이동
    try:
        page = await browser_session.get_current_page()
        await page.goto(task_info["start_url"])
        await asyncio.sleep(8)
    except Exception as e:
        print(f"  초기 페이지 이동 실패: {e}")

    output_dir = Path(f"trajectories_integrated_{model_name}")
    data_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"conversations_integrated_{model_name}").mkdir(parents=True, exist_ok=True)
    (data_dir / f"gifs_integrated_{model_name}").mkdir(parents=True, exist_ok=True)

    agent = LoggedAgentVL(
        task=full_task,
        llm=llm,
        extend_system_message=EXTEND_SYSTEM_MESSAGE,
        log_path=str(data_dir / f"training_data_integrated_{model_name}.jsonl"),
        task_index=task_id,
        save_conversation_path=str(data_dir / f"conversations_integrated_{model_name}/task_{task_id:03d}.json"),
        generate_gif=str(data_dir / f"gifs_integrated_{model_name}/task_{task_id:03d}.gif"),
        browser=browser_session,
        use_vision=True,
    )

    result = {
        "task_id": task_id, "task": task,
        "success": None, "error": None, "steps": None,
        "total_duration_seconds": None, "final_result": None,
    }

    try:
        start_time = asyncio.get_event_loop().time()
        history = await agent.run(max_steps=MAX_STEPS)
        elapsed = asyncio.get_event_loop().time() - start_time

        trajectory = parse_trajectory(task, history, elapsed)
        agent.flush_logs(success=trajectory["success"])

        output_path = output_dir / f"task_{task_id:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        result["success"] = trajectory["success"]
        result["steps"] = trajectory["total_steps"]
        result["total_duration_seconds"] = trajectory["total_duration_seconds"]
        result["final_result"] = trajectory["final_result"]

        status = "✅ 성공" if result["success"] else "❌ 실패"
        print(f"결과: {status} ({result['steps']} steps, {result['total_duration_seconds']}초)")

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        print(f"에러: {e}")
    finally:
        await browser_session.stop()

    await asyncio.sleep(SLEEP_BETWEEN_TASKS)
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gemini", "gpt"], required=True,
                        help="사용할 모델: gemini (gemini-2.5-flash) 또는 gpt (gpt-4o-mini)")
    args = parser.parse_args()

    if args.model == "gemini":
        from browser_use.llm.google.chat import ChatGoogle
        llm = ChatGoogle(model="gemini-2.5-flash")
        model_name = "gemini"
        print(f"\n🤖 Gemini 2.5 Flash")
    else:
        from browser_use.llm.openai.chat import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini")
        model_name = "gpt"
        print(f"\n🤖 GPT-4o-mini")

    print(f"총 {len(INTEGRATED_TASKS)}개 통합 시나리오 테스트 시작\n")

    Path("data/results").mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, task_info in enumerate(INTEGRATED_TASKS):
        result = await run_task(task_info, llm, model_name, i + 1, len(INTEGRATED_TASKS))
        all_results.append(result)

    success_count = sum(1 for r in all_results if r["success"])
    print(f"\n{'='*60}")
    print(f"성공: {success_count}개 / 실패: {len(all_results)-success_count}개 / 성공률: {success_count/len(all_results)*100:.1f}%")

    result_path = Path("data") / "results" / f"final_integrated_{model_name}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())