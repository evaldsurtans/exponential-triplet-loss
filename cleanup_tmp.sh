#!/bin/sh -v

ssh wn43 rm -Rf /tmp/exp_loss
ssh wn56 rm -Rf /tmp/exp_loss
ssh wn57 rm -Rf /tmp/exp_loss
ssh wn58 rm -Rf /tmp/exp_loss

ssh wn60 rm -Rf /tmp/exp_loss
ssh wn59 rm -Rf /tmp/exp_loss


ssh wn43 rm -Rf /scratch/exp_loss
ssh wn56 rm -Rf /scratch/exp_loss
ssh wn57 rm -Rf /scratch/exp_loss
ssh wn58 rm -Rf /scratch/exp_loss

ssh wn60 rm -Rf /scratch/exp_loss
ssh wn59 rm -Rf /scratch/exp_loss