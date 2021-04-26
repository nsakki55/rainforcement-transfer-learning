def _train_shared(self, max_step=None, pre_policy=None):
        """ 
        Train the transferred model for 400(= shared_max_step) 
        steps of minibatches of 32 examples.

        Args:
            max_step: Used to run extra training steps as a warm-up.
            pre_policy: If not None, is used instead of calling sample().

        For each weight update, gradients are estimated by sampling M optimizer
        from the fixed controller policy, and averaging their gradients
        computed on a batch of training data.
        """
        model = self.shared
        model.train()

        self.controller.eval()

        if max_step is None:
            max_step = self.args.shared_max_step
        else:
            max_step = max(self.args.shared_max_step, max_step)

        step = 0
        total_loss = 0
        total_acc = 0
        total_count = 0
        correct_count = 0
    
        for inputs, labels in self.train_data:
            if step > max_step:
                break

            policy = pre_policy if pre_policy else self.controller.sample(
                                            self.args.shared_num_sample)

            if self.args.cuda:
                inputs = inputs.to('cuda:0')
                labels = labels.to('cuda:0')

            optimizer = self.set_lr_optimizer(model, self.shared_optim, policy)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                loss, corrects = self.get_loss(model, inputs, labels)

                loss.backward()
                optimizer.step()

            total_loss += loss.data
            total_count += labels.size(0)
            correct_count += corrects.item()
            total_acc = correct_count / total_count

            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_shared_train(total_loss, total_acc)
                total_loss = 0
                total_acc = 0
                correct_count = 0
                total_count = 0

            step += 1
            self.shared_step += 1

    def _train_cotroller(self):
        """Fixes the shared parameters and updates the controller parameters.

        The controller is updated with a score function gradient estimator
        (i.e., REINFORCE), with the reward being valid_acc, where valid_acc
        is computed on a minibatch of validation data.

        A moving average baseline is used.

        The controller is trained for 2000 steps per epoch (i.e.,
        first (Train Shared) phase -> second (Train Controller) phase).

        Cotroller sample the policy, then train shared network by optimization
        established from policy and calculate validation accuracy after few step.
    
        """
        
        # shared networkの現在のパラメータを取得
        shared_params = self.shared.state_dict()
        self.shared.eval()

        controller_model = self.controller
        controller_model.train()

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_histroy = []
        reward_histroy = []

        total_loss = 0

        for step in range(self.args.controller_max_step):
            policy, log_probs, entropies = self.controller.sample(
                                                with_details=True)
            np_entropies = entropies.data.cpu().numpy()
            
            # 検証正解率を求めるネットワークの初期状態を共通のパラメータを用いる
            valid_model = self.build_shared_model()
            valid_model.load_state_dict(shared_params)
            valid_model.train()

            valid_optimizer = self.set_lr_optimizer(valid_model, self.shared_optim, policy)

            rewards = self.get_reward(valid_model, valid_optimizer, np_entropies)
            
            # ノイズ除去? 導入するかはENASの実装から判断する
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount) 

            reward_histroy.extend(rewards)
            entropy_histroy.extend(np_entropies)

            # REINFORCE with baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            # policy loss
            loss = -log_probs * utils.get_variable(adv, 
                                                   self.cuda, 
                                                   requires_grad=False)

            #loss.mean() でも可能、どちらを用いるか要検討
            loss = loss.sum() 

            self.controller_optim.zero_grad()
            loss.backward()

            # 勾配クリッピング, RNNの学習でよく行われるテクニック
            # https://www.madopro.net/entry/rnn_lm_on_wikipedia
            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(controller_model.parameters(),
                                                self.args.controller_grad_clip)
            self.controller_optim.step()
            
            total_loss += loss.data.item()

            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_controller_train(total_loss, 
                                                 adv_history, 
                                                 entropy_histroy,
                                                 reward_histroy,
                                                 avg_reward_base,
                                                 policy)
                reward_histroy, adv_history, entropy_histroy = [], [], []
                total_loss = 0

            self.controller_step += 1

 
    def _get_reward(self, model, optimizer, entropies):
        """Compute validation accuracy of shared model optimized from policy
        Each shared model are trained for 100 iteration and calculete validation accuracy 
        for 100 iteration validation datasets
        """

        train_loss = 0
        train_step = 0

        train_correct_count = 0
        train_total_count = 0
        # policyから作成したoptimizerを用いてsharedネットワークを学習
        logger.debug('train shared model start step: {}'.format(
            self.args.shared_reward_train_max_step))
        
        for inputs, labels in self.train_data:
            if train_step > self.args.shared_reward_train_max_step:
                break

            if self.args.cuda:
                inputs = inputs.to('cuda:0')
                labels = labels.to('cuda:0')
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                loss, corrects = self.get_loss(model, inputs, labels)
                
                loss.backward()
                optimizer.step()

            train_loss += loss.data
            train_correct_count += corrects.item()
            train_total_count += labels.size(0)

            train_step += 1
        train_acc = train_correct_count / train_total_count
        train_loss_avg =  train_loss.item() / train_step
        logger.debug(f'train shared model complete '
                     f'| loss: {train_loss_avg:.3f} | acc {train_acc:.3f}')
    
        # validation accuracyの計算
        model.eval()
        total_count = 0
        correct_count = 0
        val_loss = 0
        val_step = 0
        logger.debug('validate shared model start step {}'.format(
            self.args.shared_reward_valid_max_step))
        
        for inputs, labels in self.valid_data:
            if val_step > self.args.shared_reward_valid_max_step:
                break
            
            if self.args.cuda:
                inputs = inputs.to('cuda:0')
                labels = labels.to('cuda:0')
            
            with torch.set_grad_enabled(False):
                loss, corrects = self.get_loss(model, inputs, labels)
            
            correct_count += corrects.item()
            total_count += labels.size(0)

            val_loss += loss
            val_step += 1

        val_acc = correct_count / total_count
        val_loss_avg = val_loss.item() / val_step
        logger.debug(f'train shared model complete '
                     f'| loss: {val_loss_avg:.3f} | acc {val_acc:.3f}')

        rewards = val_acc + self.args.entropy_coeff * entropies

        return rewards

    def evaluate(self, data, model, name):
        """Evaluate on the validation set."""
        self.shared.eval()
        self.controller.eval()

        total_loss = 0
        total_count = len(data)
        correct_count = 0
        for inputs, labels in tqdm(data):
            if self.args.cuda:
                inputs = inputs.to('cuda:0')
                labels = labels.to('cuda:0')
            
            loss, corrects = self.get_loss(model, inputs, labels)
            correct_count += corrects.item()
            total_loss += len(inputs) * loss.data
        
        val_loss = total_loss.item() / total_count
        val_acc = correct_count / total_count
        self.tb.add_scalar(f'name/{name}_loss', val_loss, self.epoch)
        self.tb.add_scalar(f'name/{name}_acc', val_acc, self.epoch)
        
        logger.info(f'eval | loss: {val_loss:.2f} | acc: {val_acc:.2f}')
