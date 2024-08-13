.PHONY: build run clean

build:
	docker build -f Dockerfile.flask -t smart-box-flask .
	docker build -f Dockerfile.jupyter -t smart-box-jupyter .

run:
	docker run -p 8880:8880 -v .:/app smart-box-flask
	docker run -p 8888:8888 -v .:/app smart-box-jupyter

clean:
	@docker rm -f $$(docker ps -qa) || true
	@docker rmi -f $$(docker images -qa) || true 
