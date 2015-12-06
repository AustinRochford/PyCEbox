Vagrant.configure(2) do |config|
  config.vm.provider "virtualbox" do |v|
    v.memory = 2048
    v.cpus = 1
  end

  config.vm.box = "ubuntu/trusty64"
  config.vm.network :forwarded_port, host: 8000, guest: 8888
  config.vm.provision :shell, path: "provision.sh"
end
