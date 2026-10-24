package com.twitter.botmaker.function.collection;

import java.util.List;

import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.FunctionFailure;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode1;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    argTypes = {"List<T>"},
    arguments = {"the list"},
    returnType = "T",
    deprecated = false,
    description = "The First element of the list",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "First(List(1))"
    }
)
public class First extends FunctionNode1<Runtime, List<Object>> {
  private static final Type TP = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.listOf(TP)),
      TP);

  public First(String exprText, ImmutableList<ASTNode> children) throws SemanticCheckFailure {
    super(exprText, children);
  }

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  @Override
  public Object apply(Context<Runtime> context, List<Object> list) {
    if (list == null || list.isEmpty()) {
      throw new FunctionFailure(
          this,
          context.getStackFrames(),
          new IllegalArgumentException("First() called on empty list"));
    }
    return list.get(0);
  }
}
