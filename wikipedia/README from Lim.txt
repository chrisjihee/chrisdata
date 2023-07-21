안녕하세요. 임준호 입니다.



류지희 선임께 전화로 문의하였던 위키피디아 데이터 수집 관련 메일 드립니다.



- 일전에 보내주신 수집 프로그램은 활용하지 못하였어요. ^^;

- 아래 오픈소스를 이용해서 수집 프로그램을 별도로 개발하였어요.

(단락 별로 구분된 정보 활용 가능 api에요.)

https://pypi.org/project/Wikipedia-API/

https://github.com/martin-majlis/Wikipedia-API/



- 데이터 수집 시, api 호출 횟수가 많아서 4시간만에 block이 되었는데... 

연구팀에서 40개 ip를 이용하던 방법과 같이 수집 가능할지 문의 드려요.



- 소스코드는 간단해서 바로 쉽게 이해할게요. :)

(파일 저장은 crawl된 원문, docqa 색인에 사용할 json 2개를 저장해요.)

- 샘플 실행 스크립트: python 002.crawl_wikipedia.py --infile ./title.list --outfolder1 /Data/home/ebs_data/kowiki-20220901/result_raw/ --outfolder2 /Data/home/ebs_data/kowiki-20220901/result_json/ &> log.002.crawl_wikipedia



감사합니다.



임준호 드림. 
