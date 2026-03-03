"""
테스트 결과를 분석하는 스크립트
"""

def analyze_game_log():
    """game_log.txt를 읽어서 상세 분석"""
    try:
        with open('game_log.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("=== 게임 로그 분석 ===\n")
        print(f"총 로그 크기: {len(content)} 바이트")
        
        lines = content.split('\n')
        print(f"총 라인 수: {len(lines)}")
        
        # 키워드 검색
        keywords = {
            'summon': '소환',
            'attack': '공격',
            'damage': '피해',
            'draw': '드로우',
            'play': '플레이',
            'death': '사망',
            'trigger': '트리거'
        }
        
        print("\n주요 이벤트:")
        for keyword, korean in keywords.items():
            count = sum(1 for line in lines if keyword in line.lower())
            if count > 0:
                print(f"  {korean}: {count}번")
        
        print("\n로그 샘플 (처음 20줄):")
        for i, line in enumerate(lines[:20]):
            if line.strip():
                print(f"  {i+1}: {line[:100]}")
        
        return True
    except FileNotFoundError:
        print("game_log.txt 파일을 찾을 수 없습니다.")
        print("먼저 test_rl_env.py를 실행해주세요.")
        return False

if __name__ == "__main__":
    analyze_game_log()
