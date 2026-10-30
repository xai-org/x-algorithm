package com.twitter.botmaker.function.collection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.FunctionFailure;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode2O1;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    argTypes = {
        "String or List<OBJECT>",
        "Long",
        "[Long]"
    },
    arguments = {
        "the input string or the input list",
        "start index (inclusive) of the substring or the sublist",
        "end index (exclusive) of the substring or the sublist"
    },
    name = {"Slice", "Substring"},
    returnType = "String",
    description = "Returns a substring or a sublist",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {
        "Slice(\"teststringteststring\", 1)",
        "Substring(\"teststringteststring\", 1, 5)",
        "Slice([1, 2, 3, 4], 1)",
        "Slice([1, 2, 3, 4], 1, 2)"
    }
)
public class Slice extends FunctionNode2O1<Runtime, Object, Long, Long> {

  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.OBJECT, Type.LONG),
      ImmutableList.of(Type.LONG),
      Type.OBJECT
  );

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  public Slice(String exprText, ImmutableList<ASTNode> children) throws SemanticCheckFailure {
    super(exprText, validateType(children));
  }

  private static ImmutableList<ASTNode> validateType(ImmutableList<ASTNode> children)
      throws SemanticCheckFailure {
    Class<?> inputType = children.get(0).getReturnType().typeBase;
    if (!Object.class.equals(inputType)
      && !String.class.equals(inputType)
      && !List.class.equals(inputType)) {
      throw new SemanticCheckFailure(mkErrorMessage(inputType));
    }
    return children;
  }

  @Override
  protected Object apply(
      Context<Runtime> context, Object input, Long beginIndex, Long endIndex) {
    validateSliceBounds(context, input, beginIndex, endIndex);
    if (input instanceof String) {
      return ((String) input).substring(beginIndex.intValue(), endIndex.intValue());
    } else if (input instanceof List) {
      return Collections.unmodifiableList(new ArrayList<>(
          ((List) input).subList(beginIndex.intValue(), endIndex.intValue())));
    } else {
      throw new IllegalArgumentException(mkErrorMessage(input.getClass()));
    }
  }

  @Override
  protected Object apply(Context<Runtime> context, Object input, Long beginIndex) {
    if (input instanceof String) {
      String string = (String) input;
      return apply(context, string, beginIndex, Long.valueOf(string.length()));
    } else if (input instanceof List) {
      List list = (List) input;
      return apply(context, list, beginIndex, Long.valueOf(list.size()));
    } else {
      throw new IllegalArgumentException(mkErrorMessage(input.getClass()));
    }
  }

  private void validateSliceBounds(
      Context<Runtime> context, Object input, Long beginIndex, Long endIndex) {
    if (beginIndex == null || endIndex == null) {
      throw new FunctionFailure(
          this,
          context.getStackFrames(),
          new IllegalArgumentException("Slice() requires non-null begin and end indices"));
    }
    int begin = beginIndex.intValue();
    int end = endIndex.intValue();
    final int max;
    if (input instanceof String) {
      max = ((String) input).length();
    } else if (input instanceof List) {
      max = ((List<?>) input).size();
    } else {
      throw new IllegalArgumentException(mkErrorMessage(input.getClass()));
    }
    if (begin < 0 || end < 0 || begin > end || end > max) {
      throw new FunctionFailure(
          this,
          context.getStackFrames(),
          new IllegalArgumentException(
              String.format(
                  "Slice() indices out of range: begin=%d end=%d length=%d",
                  begin,
                  end,
                  max)));
    }
  }

  private static String mkErrorMessage(Class<?> input) {
    return String.format(
        "expecting the input to be of String or List type but received: %s for %s",
        input.getSimpleName(),
        input);
  }
}
