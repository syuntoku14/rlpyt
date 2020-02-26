# umask 0002 is to change the permission to a normal user

run_docker() {
docker run --rm -it --privileged \
	-p 6080:6080 \
	-p 8888:8888 \
	-v ~/RL_ws:/root/RL_ws \
	-v ~/rlpyt:/root/rlpyt \
	-e DISPLAY=:0 \
	--name rl \
	--shm-size 16G \
	--entrypoint "" \
	syuntoku/rl_ws:rlpyt bash -c "umask 0002 && bash"
}

run_docker_gpu() {
docker run --rm -it --privileged \
	-p 6080:6080 \
	-p 8888:8888 \
	-v ~/RL_ws:/root/RL_ws \
	-v ~/rlpyt:/root/rlpyt \
	-e DISPLAY=:0 \
	--name rl \
	--shm-size 16G \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	--gpus=all \
	--entrypoint "" \
	syuntoku/rl_ws:rlpyt bash -c "umask 0002 && bash"
}

getopts "n" OPT
case $OPT in
	n ) echo "--runtime=nvidia"
		run_docker_gpu ;;
	? )	echo "Without gpu"
		run_docker ;;
esac
