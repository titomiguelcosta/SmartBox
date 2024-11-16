.PHONY: build run clean

build:
	docker build -f Dockerfile.flask -t smart-box-flask .
	docker build -f Dockerfile.jupyter -t smart-box-jupyter .

run:
	docker run --name smart-box-flask -p 8880:8880 -v .:/app smart-box-flask  &
	docker run --name smart-box-jupyter -p 8888:8888 -v .:/app smart-box-jupyter &

restart:
	docker stop smart-box-flask
	docker start smart-box-flask

clean:
	@docker rm -f $$(docker ps -qa) || true
	@docker rmi -f $$(docker images -qa) || true
