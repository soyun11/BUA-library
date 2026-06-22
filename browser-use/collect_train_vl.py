# collect_train_vl.py
# 재구성된 테스트셋(library_train_vl.py) 기반 VL 데이터 수집
# 대출반납조회 카테고리는 자동 로그인 후 수집
# 사용법:
#   python collect_train_vl.py --model gemini 0 100

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from browser_use import BrowserSession, BrowserProfile
from logged_agent_vl import LoggedAgentVL
from dotenv import load_dotenv
from tasks.library_tasks_train import (
    ALL_TASKS as LIBRARY_TASKS,
    CATEGORY_DOMAINS as LIBRARY_CATEGORY_DOMAINS,
    LOGIN_CATEGORIES as LIBRARY_LOGIN_CATEGORIES,
    LIBRARY_BASE_CONTEXT,
)

load_dotenv()

MAX_STEPS = 30
SLEEP_BETWEEN_TASKS = 3

CNU_ID = os.getenv("CNU_ID")
CNU_PW = os.getenv("CNU_PASSWORD")


def get_llm(model_name: str):
    if model_name == "gemini":
        from browser_use.llm.google.chat import ChatGoogle
        return ChatGoogle(model="gemini-2.5-flash")
    elif model_name == "gpt":
        from browser_use.llm.openai.chat import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini")
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")


async def auto_login_library(browser_session: BrowserSession):
    try:
        page = await browser_session.get_current_page()
        await page.goto("https://library.cnu.ac.kr/login")
        await asyncio.sleep(5)
        print("도서관 아이디/비번 자동 입력 중...")
        await page.evaluate(
            f"document.querySelector(\"input[name='id']\").value = '{CNU_ID}';"
        )
        await asyncio.sleep(0.5)
        await page.evaluate(
            f"document.querySelector(\"input[name='password']\").value = '{CNU_PW}';"
        )
        await asyncio.sleep(1)
        await page.evaluate(
            "document.querySelector(\"form#login\").submit();"
        )
        await asyncio.sleep(3)
        print("✅ 도서관 자동 로그인 완료")
    except Exception as e:
        print(f"⚠️ 도서관 로그인 실패: {e}")


def get_sensitive_data(category: str) -> dict | None:
    if not CNU_ID or not CNU_PW or category not in LIBRARY_LOGIN_CATEGORIES:
        return None
    return {
        "library.cnu.ac.kr": {"x_user_id": CNU_ID, "x_user_pw": CNU_PW},
    }


def parse_trajectory(task: str, category: str, history, total_elapsed: float) -> dict:
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
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "total_steps": len(steps),
        "total_duration_seconds": history_duration or round(total_elapsed, 2),
        "steps": steps,
        "final_result": final_result,
    }


async def run_task(
    task_info: dict,
    task_index: int,
    llm,
    output_dir: Path,
    data_dir: Path,
    all_tasks: list,
    model_name: str,
) -> dict:
    raw_task = task_info["task"]
    category = task_info["category"]
    full_task = f"{LIBRARY_BASE_CONTEXT} {raw_task}"

    print(f"\n{'='*60}")
    print(f"[{task_index+1}/{len(all_tasks)}] [{category}] {raw_task[:50]}...")
    print('='*60)

    allowed_domains = LIBRARY_CATEGORY_DOMAINS.get(category, ["library.cnu.ac.kr"])
    sensitive_data = get_sensitive_data(category)

    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            allowed_domains=allowed_domains,
        )
    )

    # 대출반납조회는 자동 로그인 먼저
    if category in LIBRARY_LOGIN_CATEGORIES:
        await browser_session.start()
        await auto_login_library(browser_session)

    agent = LoggedAgentVL(
        task=full_task,
        llm=llm,
        log_path=str(data_dir / f"training_data_vl_{model_name}_train.jsonl"),
        task_index=task_index,
        save_conversation_path=str(data_dir / f"conversations_vl_{model_name}_train" / f"task_{task_index:03d}.json"),
        generate_gif=str(data_dir / f"gifs_vl_{model_name}_train" / f"task_{task_index:03d}.gif"),
        browser=browser_session,
        sensitive_data=sensitive_data,
        use_vision=True,
    )

    result = {
        "task_index": task_index,
        "task": raw_task,
        "category": category,
        "success": None,
        "error": None,
        "steps": None,
        "total_duration_seconds": None,
        "final_result": None,
    }

    try:
        start_time = asyncio.get_event_loop().time()
        history = await agent.run(max_steps=MAX_STEPS)
        elapsed = asyncio.get_event_loop().time() - start_time

        trajectory = parse_trajectory(raw_task, category, history, elapsed)
        agent.flush_logs(success=trajectory["success"])

        output_path = output_dir / f"task_{task_index:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)

        jsonl_path = data_dir / f"trajectories_vl_{model_name}_test.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

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

        error_path = output_dir / f"task_{task_index:03d}_error.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump({
                "task": raw_task, "category": category,
                "timestamp": datetime.now().isoformat(),
                "success": False, "error": str(e),
            }, f, ensure_ascii=False, indent=2)

    finally:
        await browser_session.stop()

    await asyncio.sleep(SLEEP_BETWEEN_TASKS)
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemini", "gpt"], default="gemini")
    parser.add_argument("start", type=int, nargs="?", default=0)
    parser.add_argument("end", type=int, nargs="?", default=None)
    args = parser.parse_args()

    all_tasks = LIBRARY_TASKS
    start = args.start
    end = args.end or len(all_tasks)

    output_dir = Path(f"trajectories_vl_{args.model}_train")
    data_dir = Path("data")

    for d in [
        output_dir, data_dir,
        data_dir / f"conversations_vl_{args.model}_train",
        data_dir / f"gifs_vl_{args.model}_train",
        data_dir / "results",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n🤖 모델: {args.model.upper()} (VL 테스트 데이터 수집)")
    llm = get_llm(args.model)

    print(f"총 {end - start}개 태스크 수집 시작 (index {start}~{end-1})")
    print(f"전체 태스크 수: {len(all_tasks)}개")
    print(f"저장 경로: {output_dir}/")

    if not CNU_ID:
        print("⚠️  경고: .env에 CNU_ID/CNU_PASSWORD가 없습니다.")
    else:
        print(f"🔑 로그인 계정: {CNU_ID}")

    all_results = []
    for i in range(start, end):
        result = await run_task(
            all_tasks[i], i, llm, output_dir, data_dir, all_tasks, args.model,
        )
        all_results.append(result)

        if len(all_results) % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(data_dir / "results" / f"progress_vl_{args.model}_train_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"💾 중간 저장 완료 ({len(all_results)}개)")

    print(f"\n{'='*60}")
    success_count = sum(1 for r in all_results if r["success"])
    fail_count = len(all_results) - success_count
    print(f"성공: {success_count}개 / 실패: {fail_count}개 / 성공률: {success_count/len(all_results)*100:.1f}%")

    with open(data_dir / "results" / f"final_vl_{args.model}_train.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: data/results/final_vl_{args.model}_train.json")


if __name__ == "__main__":
    asyncio.run(main())