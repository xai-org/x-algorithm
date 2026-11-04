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
import com.twitter.botmaker.function.FunctionNode2;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    argTypes = {
        "List<T>",
        "Long"
    },
    arguments = {
        "the list",
        "the index"
    },
    returnType = "T",
    deprecated = false,
    description = "The element at the given index.",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "ElementAtIndex(List(\"hello\", 42), 0)"
    }
)
public class ElementAtIndex extends FunctionNode2<Runtime, List<Object>, Long> {

  private static final Type TP = Type.newGenericType();
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.listOf(TP), Type.LONG),
      TP
  );

  public ElementAtIndex(String exprText, ImmutableList<ASTNode> children)
      throws SemanticCheckFailure {
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
  protected Object apply(Context<Runtime> context, List<Object> list, Long index) {
    if (list == null) {
      throw new FunctionFailure(
          this,
          context.getStackFrames(),
          new IllegalArgumentException("ElementAtIndex() called on null list"));
    }
    if (index == null) {
      throw new FunctionFailure(
          this,
          context.getStackFrames(),
          new IllegalArgumentException("ElementAtIndex() requires a non-null index"));
    }
    int i = index.intValue();
    if (i < 0 || i >= list.size()) {
      throw new FunctionFailure(
          this,
          context.getStackFrames(),
          new IllegalArgumentException(
              String.format(
                  "ElementAtIndex() index out of range: index=%d size=%d",
                  i,
                  list.size())));
    }
    return list.get(i);
  }

}
